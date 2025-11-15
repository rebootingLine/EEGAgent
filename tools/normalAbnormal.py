import torch
import torch.nn as nn
import math
from .localModels.net import RoPETransformer
from .register import function_register
from .registerData import getRegisteredData
import scipy.signal as signal
import numpy as np

class NormalAbnormalEEG(nn.Module):
    def __init__(self, window, n_dim, n_head, n_layer) -> None:
        super().__init__()
        self.window = window
        self.n_dim = n_dim
        self.n_layer = n_layer
        self.conv = nn.Sequential(
            nn.Conv1d(22, 22, 75, 1),
            nn.GroupNorm(22, 22),
            nn.GELU(approximate='tanh'),
            nn.MaxPool1d(15, 15),
            nn.Conv1d(22, n_dim, 15, 1),
            nn.GroupNorm(n_dim, n_dim),
            nn.GELU(approximate='tanh'),
            nn.MaxPool1d(5, 5),
            nn.AdaptiveAvgPool1d(1)
        )
        self.global_token = nn.Parameter(torch.normal(mean=0.0, std=1/math.sqrt(n_dim), size=(1, 1, n_dim)))
        self.timeAttn = nn.ModuleList([RoPETransformer(n_dim, n_head) for _ in range(self.n_layer)])
        self.ln = nn.LayerNorm(n_dim)
        self.lm_head = nn.Linear(n_dim, 2)
        
    def forward(self, x, mask):
        B, C, T = x.shape
        W = self.window
        x = x.view(B, C, T//W, W).transpose(1, 2).flatten(0, 1) # (B*T//W, C, W)
        x = self.conv(x).squeeze(-1) # (B*T//W, E)
        x = x.reshape(B, T//W, -1)
        x = torch.cat([self.global_token.expand(B, 1, x.shape[-1]), x], dim=1)
        for layer in self.timeAttn:
            x, _ = layer(x, mask)
        x = self.ln(x)
        x = x[:,0,:]
        out = self.lm_head(x)
        return out
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 1 / math.sqrt(self.n_dim)
            if hasattr(module, 'SCALE'):
                std = (2 * self.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

import os
model_normalEEG = NormalAbnormalEEG(10*100, 64, 4, 8)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "localModels", "normal&abnormal.pth")
model_normalEEG.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True)) # 这里会自动加载到历史cuda设备上
model_normalEEG.eval()

@function_register.register(
    description="Analyze the probability of pathological normality and abnormality in the entire EEG record.",
    parameters={}, 
    returns={
        "type": "Dict[str, float]",
        "description": "Probability of pathological normality and abnormality"
    }
)
def normalAbnormalModel(config):
    target_fs = 100
    desired_length = 120000  # 20 minutes × 100 Hz
    window_size = 1000        # 10 seconds per window
    num_windows = desired_length // window_size  

    data = getRegisteredData()  # shape: (C, T)
    C, T = data.shape
    original_fs = config['fs']

    new_T = int(T * target_fs / original_fs)
    resampled_data = signal.resample(data, num=new_T, axis=1)  # (C, new_T)

    current_T = resampled_data.shape[1]
    if current_T < desired_length:
        pad_len = desired_length - current_T
        padded_data = np.pad(resampled_data, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
    else:
        padded_data = resampled_data[:, :desired_length]

    # Step 3: construct mask
    mask = torch.ones((1, num_windows+1), dtype=torch.int32)
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = padded_data[:, start:end]
        if np.allclose(window, 0): # 默认阈值是1e-8
            mask[0, i+1] = 0

    # Step 4: model inferring
    data_tensor = torch.tensor(padded_data, dtype=torch.float32).unsqueeze(0)  # (1, C, T)
    logits = model_normalEEG(data_tensor, mask) 
    probs = torch.softmax(logits, dim=1)         

    # Step 5: return prob
    return {
        "normal Probability": round(probs[0, 0].item(),2),
        "abnormal Probability": round(probs[0, 1].item(),2)
    }
