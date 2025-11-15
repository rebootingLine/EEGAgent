import os
import mne
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# label mapping
LABEL_MAP = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}

class SleepEDFDataset:
    def __init__(self, folder_path, eeg_channels=['EEG Fpz-Cz', 'EEG Pz-Oz'], epoch_length=30):
        self.folder_path = folder_path
        self.eeg_channels = eeg_channels
        self.epoch_length = epoch_length
        self.subject_map = {}  # subject_id -> list of (psg,hyp)
        self._build_file_pairs()

    def _build_file_pairs(self):
        files = os.listdir(self.folder_path)
        psg_files = sorted([f for f in files if f.endswith('-PSG.edf')])
        hyp_files = sorted([f for f in files if f.endswith('-Hypnogram.edf')])

        for psg in psg_files:
            base_prefix = psg.split('-PSG.edf')[0][:-1]
            matches = [h for h in hyp_files if h.startswith(base_prefix)]
            if matches:
                hyp_file = matches[0]
                subject_id = psg[:6]
                if subject_id not in self.subject_map:
                    self.subject_map[subject_id] = []
                self.subject_map[subject_id].append((psg, hyp_file))
            else:
                print(f"Warning: Hypnogram file not found for {psg}, skipping...")

    def _get_sleep_range(self, annotations, raw, extend_wake=1800):
        # find all wake
        wake_onsets = [ann['onset'] for ann in annotations if ann['description'] == 'Sleep stage W']
        wake_durations = [ann['duration'] for ann in annotations if ann['description'] == 'Sleep stage W']

        if len(wake_onsets) < 2:
            sleep_start = 0
            sleep_end = annotations[-1]['onset'] + annotations[-1]['duration']
        else:
            gaps = [wake_onsets[i+1] - (wake_onsets[i] + wake_durations[i]) for i in range(len(wake_onsets)-1)]
            max_gap_idx = np.argmax(gaps)
            sleep_start = wake_onsets[max_gap_idx] + wake_durations[max_gap_idx]
            sleep_end = wake_onsets[max_gap_idx+1]

        # expand extend_wake second
        sleep_start = max(0, sleep_start - extend_wake)
        sleep_end = sleep_end + extend_wake

        total_end = annotations[-1]['onset'] + annotations[-1]['duration']
        sleep_end = min(sleep_end, total_end, raw.times[-1])
        return sleep_start, sleep_end

    def _load_file_pair(self, psg_file, hyp_file):
        raw = mne.io.read_raw_edf(os.path.join(self.folder_path, psg_file), preload=True, verbose=False)
        raw.pick(picks=self.eeg_channels)

        annot_obj = mne.read_annotations(os.path.join(self.folder_path, hyp_file))
        raw.set_annotations(annot_obj)
        annotations = annot_obj

        sleep_start, sleep_end = self._get_sleep_range(annotations, raw)

        raw_sleep = raw.copy().crop(tmin=sleep_start, tmax=sleep_end)

        sf = raw_sleep.info['sfreq']
        samples_per_epoch = int(self.epoch_length * sf)
        data = raw_sleep.get_data()
        n_epochs = data.shape[1] // samples_per_epoch

        eeg_epochs = np.array([
            data[:, i*samples_per_epoch:(i+1)*samples_per_epoch]
            for i in range(n_epochs)
        ])

        epoch_labels = []
        for i in range(n_epochs):
            start_sec = sleep_start + i * self.epoch_length
            idx = np.where(
                (np.array([ann['onset'] for ann in annotations]) <= start_sec) &
                (start_sec < np.array([ann['onset'] + ann['duration'] for ann in annotations]))
            )[0]
            if len(idx) > 0:
                label = LABEL_MAP.get(annotations[idx[0]]['description'], -1)
            else:
                label = -1
            epoch_labels.append(label)

        epoch_labels = np.array(epoch_labels)
        valid_idx = epoch_labels != -1
        return eeg_epochs[valid_idx], epoch_labels[valid_idx]

    def load_dataset(self, test_size=0.3, random_state=42):
        subjects = list(self.subject_map.keys())
        train_subjects, test_subjects = train_test_split(
            subjects, test_size=test_size, random_state=random_state
        )

        def _load_subjects(subject_list):
            epochs_list, labels_list, files_list = [], [], []
            subject_files = []
            for sub in tqdm(subject_list):
                for i, (psg_file, hyp_file) in enumerate(self.subject_map[sub]):
                    eeg_epochs, labels = self._load_file_pair(psg_file, hyp_file)
                    epochs_list.append(eeg_epochs)
                    labels_list.append(labels)
                    if i == 0:
                        subject_files.append([psg_file, hyp_file])
            X = np.concatenate(epochs_list, axis=0)
            y = np.concatenate(labels_list, axis=0)
            subject_files = np.array(subject_files)
            return X, y, subject_files
        
        X_test, y_test, file_test = _load_subjects(test_subjects)
        X_train, y_train, file_train = _load_subjects(train_subjects)
        

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        return X_train, y_train, X_test, y_test, file_train, file_test

    def save_dataset(self, save_folder, X_train, y_train, X_test, y_test, file_train, file_test):
        os.makedirs(save_folder, exist_ok=True)
        torch.save(X_train, os.path.join(save_folder, 'X_train.pt'))
        torch.save(y_train, os.path.join(save_folder, 'y_train.pt'))
        torch.save(X_test, os.path.join(save_folder, 'X_test.pt'))
        torch.save(y_test, os.path.join(save_folder, 'y_test.pt'))
        np.save(os.path.join(save_folder, 'file_train.npy'), file_train)
        np.save(os.path.join(save_folder, 'file_test.npy'), file_test)
        print(f"Dataset saved to {save_folder}")


folder = "./sleep-cassette"
save_folder = "./data"

dataset = SleepEDFDataset(folder)
X_train, y_train, X_test, y_test, file_train, file_test = dataset.load_dataset()
dataset.save_dataset(save_folder, X_train, y_train, X_test, y_test, file_train, file_test)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Example test set file pairs:", file_test[:5])
