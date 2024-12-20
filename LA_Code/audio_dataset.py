import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
from scipy.signal import convolve

class AudioDataset(Dataset):
    def __init__(self, flac_folder, labels_file, transform=None):
        self.flac_folder = flac_folder
        self.labels = {} 
        self._load_labels(labels_file)
        self.transform = transform
        self.filenames = [f for f in os.listdir(flac_folder) if f.endswith(".flac")]

    def _load_labels(self, label_file_path):
        # Read the label file and populate the dictionary
        with open(label_file_path, "r") as file:
            for line in file:
                parts = line.strip().split()  # Split the line into elements
                if len(parts) >= 5:  # Ensure there are enough elements in the line
                    audioname = parts[1]  # The second element
                    label = 1 if parts[4] == 'spoof' else 0  # 1 for spoof, 0 for bonafide
                    self.labels[audioname] = label

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        audioname = filename.split('.')[0]
        #print(audioname)
        file_path = os.path.join(self.flac_folder, filename)
        
        # Load the audio data
        audio_data, _ = librosa.load(file_path, sr=None)  # sr=None keeps original sampling rate
        # Extract features (chromatic derivatives)
        features = self.feature_extraction(audio_data)

        features_tensor = torch.tensor(features, dtype=torch.float32)
        # Get the label
        label = self.labels.get(audioname, None)
        if label is None:
            raise ValueError(f"Label not found for {audioname}")
        label_tensor = torch.tensor(label, dtype=torch.float32)

        sample = {'features': features_tensor, 'label': label_tensor}
        #print('Got sample')
        if self.transform:
            sample = self.transform(sample)
            
        return sample

    def feature_extraction(self, audio_data):
        # print('Computing Feature Extraction')
        chroma_features = self.get_CDs(audio_data)
        return chroma_features

    def get_CDs(self, sig_sec):
        deg = 47
        imp = np.zeros((129, deg + 1))
        lh = len(imp)
        
        # Load impulse responses (if needed per sample)
        for k in range(deg + 1):
            imp[:, k] = np.flipud(np.loadtxt(f'/kaggle/working/FYP_CDs/LA_Code/FilterBank/legendre_0.885_0.985_1.00_129_{k}.txt')) / 1e15
        
        N = len(sig_sec) // 2
        sig_in = np.concatenate([np.zeros(lh // 2 + 1), sig_sec, np.zeros(lh // 2 + 1)])
        
        CD = np.zeros((len(sig_in) + len(imp) - 1, deg + 1))

        for k in range(deg + 1):
            CD[:, k] = convolve(sig_in, imp[:, k])

        CD = CD[lh:-lh, :]
        CD = CD[::24]

        # Ensure CD has 5147 rows by padding with zeros if necessary
        # target_rows = 8792
        # current_rows, cols = CD.shape
        # if current_rows < target_rows:
        #     padding = np.zeros((target_rows - current_rows, cols))
        #     CD = np.vstack((CD, padding))
        return CD
