import torch
import torch.nn as nn
from .register import function_register
from .registerData import getRegisteredData
import scipy.signal as signal

class SleepStage(nn.Module):
    def __init__(self, n_channels=2, n_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 24, kernel_size=31, stride=1),
            nn.LayerNorm(170),
            nn.ReLU(),
            nn.AvgPool1d(5),
            nn.Conv1d(24, 64, kernel_size=5, stride=1),
            nn.LayerNorm(30),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.feature_dim = 64  
        
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(32*2, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
            )
        
        # self.apply(self.init_weights)

    def forward(self, x):
        # x: [batch, n_channels, time_samples]
        batch_size, n_channels, total_len = x.shape
        sec_len = total_len // 15
        x = x.view(batch_size, 15, n_channels, sec_len).flatten(0, 1)

        x = self.cnn(x)
        x = x.reshape(batch_size, 15, -1)
        
        # GRU
        out, _ = self.gru(x)
        out = out[:, -1, :]
        logits = self.fc(out)
        return logits
    


import os
model_sleepEEG = SleepStage()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "localModels", "sleepStage.pth")
model_sleepEEG.load_state_dict(
    torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
)
model_sleepEEG.eval()

# -------------------------
# Label mapping
# -------------------------
SLEEP_LABELS = ["W", "N1", "N2", "N3", "R"]

# -------------------------
# Tool registration
# -------------------------
@function_register.register(
    description=(
        "Predict 5-class sleep stages for a given EEG segment. "
        "The input segment should be at least 30 seconds long. "
        "This tool divides the segment into non-overlapping 30-second epochs, and returns the probability"
        "for each of the 5 sleep stages ('W','N1','N2','N3','R') for each epoch. "
    ),
    parameters=[
        {
            "name": "start",
            "type": "int",
            "description": "Start time of the EEG segment in seconds."
        },
        {
            "name": "end",
            "type": "int",
            "description": "End time of the EEG segment in seconds."
        }
    ],
    returns={
        "type": "List[Dict[str, any]]",
        "description": (
            "A list of dictionaries, one per predicted segment. "
            "Each dictionary contains 'duration' as the time window string, "
            "and 'Prob' which is a dictionary of sleep stage probabilities, e.g., "
            "{'W': 0.1, 'N1': 0.2, 'N2': 0.5, 'N3': 0.1, 'R': 0.1}."
        )
    }
)
def sleepStageModel(start: int, end: int, config, segment_length=30):
    if end - start < segment_length:
        return [{"warning": f"Data length {end-start:.2f}s is too short for a {segment_length}s prediction"}]

    data = getRegisteredData(start, end, config)  # [C, T]
    fs_target = config['fs']
    duration_sec = end - start
    # Resample data to target sampling rate
    data = signal.resample(data, num=int(duration_sec * fs_target), axis=1)

    n_samples = data.shape[1]
    seg_samples = segment_length * fs_target
    n_segments = n_samples // seg_samples

    results = []
    for i in range(n_segments):
        start_idx = i * seg_samples
        end_idx = start_idx + seg_samples
        seg = data[:, start_idx:end_idx]
        x_tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)  # [1, C, T]

        with torch.no_grad():
            logits = model_sleepEEG(x_tensor)
            probs = torch.softmax(logits, dim=1)

        prob_dict = {label: round(probs[0, idx].item(), 2) for idx, label in enumerate(SLEEP_LABELS)}

        start_time = start + start_idx / fs_target
        end_time = start + end_idx / fs_target
        results.append({
            "duration": f"{start_time:.2f}s-{end_time:.2f}s",
            "Prob": prob_dict
        })

    return results
