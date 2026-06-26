import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class ICUDataset(Dataset):
    def __init__(self, csv_file, max_len=48):
        """
        Loads ICU data and groups rows by patient to form sequences.
        """
        self.df = pd.read_csv(csv_file)
        self.max_len = max_len
        self.features = ['heart_rate', 'spo2', 'systolic_bp', 'temperature', 'creatinine']
        
        # Group time series entries by unique patient IDs
        self.patient_groups = [group for _, group in self.df.groupby('patient_id')]

    def __len__(self):
        return len(self.patient_groups)

    def __getitem__(self, idx):
        patient_data = self.patient_groups[idx]
        
        # Sort by timestep to preserve clinical timeline
        patient_data = patient_data.sort_values('timestep')
        seq_data = patient_data[self.features].values
        seq_len = min(len(seq_data), self.max_len)
        
        # Initialize padded array and mask arrays
        padded_seq = np.zeros((self.max_len, len(self.features)), dtype=np.float32)
        padding_mask = np.zeros(self.max_len, dtype=np.float32) # 0 means real data, 1 means padding
        
        # Populate arrays up to actual sequence length
        padded_seq[:seq_len, :] = seq_data[:seq_len, :]
        padding_mask[seq_len:] = 1.0  
        
        return {
            'x': torch.tensor(padded_seq),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool)
        }

if __name__ == "__main__":
    # Quick sanity test execution
    dataset = ICUDataset('data/hospital_1_data.csv')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Fetch one sample batch
    sample_batch = next(iter(dataloader))
    print("--- Pipeline Test Successful ---")
    print("Batch Features Shape (Batch, Timesteps, Features):", sample_batch['x'].shape)
    print("Batch Padding Mask Shape (Batch, Timesteps):", sample_batch['padding_mask'].shape)
