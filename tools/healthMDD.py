import torch
import torch.nn as nn
from .register import function_register
from .registerData import getRegisteredData
import scipy.signal as signal

class healthMDD(nn.Module):
    def __init__(self, n_channels=19, n_samples=1280, hidden=64, p=0.):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 24, 75, 1),
            nn.LayerNorm(1206),
            nn.ReLU(),
            nn.AvgPool1d(5, 5),
            nn.Dropout(p),
            nn.Conv1d(24, 48, 15, 1),
            nn.LayerNorm(227),
            nn.ReLU(),
            nn.AvgPool1d(5, 5),
            nn.Dropout(p),
            nn.Conv1d(48, hidden, 5, 1),
            nn.LayerNorm(41),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        x = self.conv(x).flatten(1) # (32, 48, 49)
        return self.fc(x)
    

import os
model_normalEEG = healthMDD()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "localModels", "health&MDD.pth")
model_normalEEG.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True)) 
model_normalEEG.eval()

@function_register.register(
    description=(
        "Analyze the probability of health and MDD in the EEG data between [start, end] seconds. "
        "Minimum segment length is 5 seconds. "
        "If the interval is longer than 5 seconds, the function splits it into consecutive 5-second segments "
        "and returns a list of probabilities for each segment. "
        "Probabilities can be averaged to get a subject-level assessment."
    ),
    parameters=[{
            "name": "start",
            "type": "int",
            "description": "Start time in seconds"
        },
        {
            "name": "end",
            "type": "int",
            "description": "End time in seconds"
        }, 
    ],
    returns={
        "type": "List[Dict[str, Any]]",
        "description": "List of dictionaries with segment duration and probability of Health and MDD"
    }
)
def healthMDDModel(start: int, end: int, config, segment_length=5):
    if end - start < segment_length:
        return [{"warning": f"data length {end-start:.2f}s < {segment_length}s"}]

    data = getRegisteredData(start, end, config)  # shape: [C, T]
    fs_target = config['fs']
    duration_sec = end - start
    data = signal.resample(data, num=int(duration_sec * config['fs']), axis=1)

    n_samples = data.shape[1]
    seg_samples = segment_length * fs_target
    n_segments = n_samples // seg_samples  # 只取完整片段

    results = []

    for i in range(n_segments):
        start_idx = i * seg_samples
        end_idx = start_idx + seg_samples
        seg = data[:, start_idx:end_idx]

        x_tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = model_normalEEG(x_tensor)
            probs = torch.softmax(logits, dim=1)
            prob_H = round(probs[0, 0].item(), 2)
            prob_MDD = round(probs[0, 1].item(), 2)

        start_time = start + start_idx / fs_target
        end_time = start + end_idx / fs_target
        results.append({
            "duration": f"{start_time:.2f}s-{end_time:.2f}s",
            "Prob": {"H": prob_H, "MDD": prob_MDD}
        })

    return results