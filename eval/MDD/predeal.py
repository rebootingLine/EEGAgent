import os
import numpy as np
import torch
import mne
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RAW_FOLDER = "./raw"
SAVE_FOLDER = "./processed"
EPOCH_LENGTH = 5          
STRIDE = 2.5              
SFREQ = 256               
EEG_CHANNELS = [
    'EEG Fp1-LE', 'EEG F3-LE', 'EEG C3-LE', 'EEG P3-LE', 'EEG O1-LE',
    'EEG F7-LE', 'EEG T3-LE', 'EEG T5-LE', 'EEG Fz-LE', 'EEG Fp2-LE',
    'EEG F4-LE', 'EEG C4-LE', 'EEG P4-LE', 'EEG O2-LE', 'EEG F8-LE',
    'EEG T4-LE', 'EEG T6-LE', 'EEG Cz-LE', 'EEG Pz-LE'
]

os.makedirs(SAVE_FOLDER, exist_ok=True)

all_files = [f for f in os.listdir(RAW_FOLDER) if f.endswith('.edf') and 'EC' in f]
print(len(all_files))
subjects = {}

for f in all_files:
    prefix = f.split(' ')[0]  # 'H' or 'MDD'
    s_number = f.split(' ')[1] 
    subject_id = f"{prefix}_{s_number}"
    if subject_id not in subjects:
        subjects[subject_id] = f

subject_ids = list(subjects.keys())

train_ids, test_ids = train_test_split(subject_ids, test_size=0.3, random_state=42)

def slice_eeg(raw, epoch_length=EPOCH_LENGTH, stride=STRIDE):
    sfreq = int(raw.info['sfreq'])
    data = raw.get_data(picks=EEG_CHANNELS)
    n_samples = data.shape[1]
    step = int(stride * sfreq)
    epoch_size = int(epoch_length * sfreq)
    
    slices = []
    for start in range(0, n_samples - epoch_size + 1, step):
        slices.append(data[:, start:start+epoch_size])
    return np.array(slices)


def process_subject(subject_id, file_name):
    label = 0 if subject_id.startswith("H") else 1
    raw = mne.io.read_raw_edf(os.path.join(RAW_FOLDER, file_name), preload=True, verbose=False)
    raw.pick(EEG_CHANNELS)
    slices = slice_eeg(raw)
    labels = np.full(len(slices), label, dtype=np.int64)
    return slices, labels


X_train_list, y_train_list = [], []
for sid in tqdm(train_ids, desc="Processing train subjects"):
    slices, labels = process_subject(sid, subjects[sid])
    X_train_list.append(slices)
    y_train_list.append(labels)

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
torch.save((X_train, y_train), os.path.join(SAVE_FOLDER, "train.pt"))


X_test_list, y_test_list = [], []
for sid in tqdm(test_ids, desc="Processing test subjects"):
    slices, labels = process_subject(sid, subjects[sid])
    X_test_list.append(slices)
    y_test_list.append(labels)

X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)
torch.save((X_test, y_test), os.path.join(SAVE_FOLDER, "test.pt"))


with open(os.path.join(SAVE_FOLDER, "train_subjects.txt"), 'w') as f:
    f.write("\n".join(train_ids))

with open(os.path.join(SAVE_FOLDER, "test_subjects.txt"), 'w') as f:
    f.write("\n".join(test_ids))

print("predeal success, saved at", SAVE_FOLDER)
