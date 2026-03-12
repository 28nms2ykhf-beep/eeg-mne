# data_loader.py
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold

def load_labels(csv_path):
    """0 is left hemisphere, 1 is right hemisphere"""
    df = pd.read_csv(csv_path)
    df = df[['patient', 'session', 'position']]
    df = df.dropna(subset=['patient', 'session', 'position'])
    label_dict = {}
    for _, row in df.iterrows():
        patient_raw = str(row['patient']).strip()
        try:
            patient_num = str(int(float(patient_raw)))
        except ValueError:
            patient_num = patient_raw
        patient_folder = f"patient_{patient_num}"
        session_raw = str(row['session']).strip()
        if session_raw.startswith('ses-'):
            session_folder = session_raw
        else:
            session_folder = f"ses-{session_raw}"
        pos = str(row['position']).strip().upper()
        label = 0 if pos == 'L' else 1
        label_dict[(patient_folder, session_folder)] = label
    print(f"Loaded {len(label_dict)} (patient, session) labels.")
    left = sum(1 for v in label_dict.values() if v == 0)
    right = sum(1 for v in label_dict.values() if v == 1)
    print(f"  Left: {left}, Right: {right}")
    return label_dict


class EEGDataset(Dataset):
    def __init__(self, data_root, label_dict, samples_per_session=None, allowed_keys=None):
        self.data_root = data_root
        self.label_dict = label_dict
        self.samples_per_session = samples_per_session
        self.allowed_keys = set(allowed_keys) if allowed_keys else None
        self.samples = []
        self._collect_samples()

    def _collect_samples(self):
        patients_found = 0
        for patient in os.listdir(self.data_root):
            patient_path = os.path.join(self.data_root, patient)
            if not os.path.isdir(patient_path):
                continue
            patients_found += 1
            for session in os.listdir(patient_path):
                session_path = os.path.join(patient_path, session)
                if not os.path.isdir(session_path):
                    continue
                key = (patient, session)
                if self.allowed_keys and key not in self.allowed_keys:
                    continue
                if key not in self.label_dict:
                    continue
                label = self.label_dict[key]
                files = [f for f in os.listdir(session_path) if f.endswith('.pt')]
                if not files:
                    continue
                if self.samples_per_session and len(files) > self.samples_per_session:
                    files = random.sample(files, self.samples_per_session)
                for fname in files:
                    fpath = os.path.join(session_path, fname)
                    self.samples.append((fpath, label, patient, session))
        print(f"Total samples collected: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label, patient, session = self.samples[idx]
        try:
            eeg = torch.load(fpath).float()
            if eeg.dim() == 2:
                if eeg.shape[0] == 18:
                    eeg = eeg.unsqueeze(0)
                elif eeg.shape[1] == 18:
                    eeg = eeg.transpose(0, 1).unsqueeze(0)
                elif eeg.numel() == 18 * 4096:
                    eeg = eeg.view(1, 18, 4096)
                else:
                    raise ValueError(f"tensor shape {eeg.shape} invalid")
            elif eeg.dim() == 3:
                if eeg.shape[2] == 18:
                    eeg = eeg.transpose(1, 2)
                if eeg.shape[0] != 1:
                    eeg = eeg[:1]
            else:
                raise ValueError(f"Dimension {eeg.dim()} invalid")
            return eeg, label, patient, session
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            return torch.zeros(1, 18, 4096), -1, patient, session


def collate_fn(batch):
    batch = [b for b in batch if b[1] != -1]
    if len(batch) == 0:
        return None
    eegs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    patients = [b[2] for b in batch]
    sessions = [b[3] for b in batch]
    return eegs, labels, patients, sessions


def split_patients(label_dict, test_ratio=0.2, val_ratio=0.1, seed=42, n_splits=None):
    session_keys = list(label_dict.keys())
    session_labels = list(label_dict.values())

    if n_splits is None:
        train_val_keys, test_keys, train_val_labels, _ = train_test_split(
            session_keys, session_labels, test_size=test_ratio,
            stratify=session_labels, random_state=seed
        )
        val_ratio_adj = val_ratio / (1 - test_ratio)
        train_keys, val_keys, _, _ = train_test_split(
            train_val_keys, train_val_labels, test_size=val_ratio_adj,
            stratify=train_val_labels, random_state=seed
        )
        print(
            f"Train sessions: {len(train_keys)} (L: {sum(1 for k in train_keys if label_dict[k] == 0)}, R: {sum(1 for k in train_keys if label_dict[k] == 1)})")
        print(
            f"Val sessions: {len(val_keys)} (L: {sum(1 for k in val_keys if label_dict[k] == 0)}, R: {sum(1 for k in val_keys if label_dict[k] == 1)})")
        print(
            f"Test sessions: {len(test_keys)} (L: {sum(1 for k in test_keys if label_dict[k] == 0)}, R: {sum(1 for k in test_keys if label_dict[k] == 1)})")
        return train_keys, val_keys, test_keys
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(session_keys, session_labels)):
            train_sessions = [session_keys[i] for i in train_idx]
            val_sessions = [session_keys[i] for i in val_idx]
            print(
                f"Fold {fold_idx+1}: train sessions {len(train_sessions)} (L: {sum(1 for s in train_sessions if label_dict[s] == 0)}, R: {sum(1 for s in train_sessions if label_dict[s] == 1)})")
            print(
                f"val sessions {len(val_sessions)} (L: {sum(1 for s in val_sessions if label_dict[s] == 0)}, R: {sum(1 for s in val_sessions if label_dict[s] == 1)})")
            folds.append((train_sessions, val_sessions))
        return folds


def weights_for_sampling(dataset):
    labels = [s[1] for s in dataset.samples]
    class_counts = np.bincount(labels, minlength=2)
    total = sum(class_counts)
    weights = total / (2 * class_counts)
    return torch.tensor(weights, dtype=torch.float32)
