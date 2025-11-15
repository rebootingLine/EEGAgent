import torch
import torch.nn as nn
from .localModels.net import RoPETransformer
from .registerData import getRegisteredData
from .localModels.vote import multi_model_predict
from tools import function_register

def compute_stft_spectrogram(x, n_fft=256, hop_length=128):
    """
    x: Tensor of shape (B, 22, T=2560)
    returns: Tensor of shape (B, 22, F=n_fft//2+1, T'=~20)
    """
    B, C, T = x.shape
    x = x.view(B * C, T)
    # STFT: (B*C, F, T') with complex numbers
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length,
        return_complex=True, window=torch.hann_window(n_fft, device=x.device)
    )
    mag = spec.abs()
    mag = mag.view(B, C, *mag.shape[1:])  # (B, 22, F, T')
    
    return mag  # shape: (B, 22, F, T')

class slowSeizureBckgEEG(nn.Module):
    def __init__(self, cls=3, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Sequential(
            nn.Conv1d(22, 22, kernel_size=3, stride=1, padding=1),
            nn.GELU(approximate='tanh'),
            nn.GroupNorm(22, 22),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(22, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(approximate='tanh'),
            nn.GroupNorm(embed_dim, embed_dim),
            nn.MaxPool1d(4, 4),
            nn.AdaptiveAvgPool1d(1)
        )

        self.attn_layers = nn.ModuleList([
            RoPETransformer(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(2)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, embed_dim))
        self.fc = nn.Linear(embed_dim, cls)
        self.apply(self._init_weights)

    def forward(self, x):
        B, _, _ = x.shape
        x = compute_stft_spectrogram(x).permute(0,3,1,2).flatten(0, 1) # B*T, 22, F
        x = self.conv(x).squeeze(2).reshape(B, -1, self.embed_dim)
        for layer in self.attn_layers:
            x, _  = layer(x)  # v = x
        x = self.pool(x).squeeze(1)  # [B, C]

        x = self.fc(x)
        return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.08)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.08)


import os
device = 'cpu'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
slowSeizBckgEEGModel_paths = [
    os.path.join(CURRENT_DIR, "localModels", "slow&seiz&bckg1.pth"),
    os.path.join(CURRENT_DIR, "localModels", "slow&seiz&bckg2.pth"),
    os.path.join(CURRENT_DIR, "localModels", "slow&seiz&bckg3.pth"),
    os.path.join(CURRENT_DIR, "localModels", "slow&seiz&bckg4.pth"),
    os.path.join(CURRENT_DIR, "localModels", "slow&seiz&bckg5.pth")
]


slowSeizBckgModels = []
for i, path in enumerate(slowSeizBckgEEGModel_paths):
    model = slowSeizureBckgEEG()
    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    slowSeizBckgModels.append(model)

@function_register.register(
    description="The probability that the data analyzed between [start, end] seconds contains slow waves, epilepsy, and background activity.",
    parameters=[
        {
            "name": "start",
            "type": "int",
            "description": "Start time in seconds."
        },
        {
            "name": "end",
            "type": "int",
            "description": "End time in seconds."
        }
    ],
    returns={
        "type": "List[Dict]",
        "description": "A list of dictionaries. Each dictionary represents 10 seconds of analyzed data."
    }
)
def slowSeizBckgModel_TenSeconds(start: int, end: int, config):
    fs = config['fs']
    N = (end- start) // 10
    data = getRegisteredData(start, start+10*N, config) # shape: (C, T)

    infos = []
    for i in range(N):
        start_idx = i * 10 * fs
        end_idx = (i + 1) * 10 * fs
        x = data[:, start_idx:end_idx]
        info = {}
        info['duration'] = f"{start + i * 10}s-{start + (i + 1) * 10}s"

        data_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, C, T)
        probs = multi_model_predict(slowSeizBckgModels, data_tensor)

        info['Prob'] = {
            "bckg": round(probs[0].item(),2),
            "slow": round(probs[1].item(),2),
            "seiz": round(probs[2].item(),2),
        }
        infos.append(info)
    
    return infos