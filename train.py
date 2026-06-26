import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.pipeline import ICUDataset
from models.transformer import ICUTransformerFM

def pretrain_local_hospital(csv_path, epochs=3, batch_size=16, lr=0.001):
    """
    Trains the ICU Foundation Model on a local hospital dataset 
    using a self-supervised Masked Feature Reconstruction loss.
    """
    # 1. Setup Data Pipeline
    dataset = ICUDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Initialize Model, Optimizer, and Loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ICUTransformerFM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction='none') # Compute element-wise loss
    
    model.train()
    print(f"Starting local pre-training on {csv_path} using {device}...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            x = batch['x'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            
            # Create a random boolean mask covering roughly 15% of the data matrix
            # This simulates missing lab tests or clinical observations
            mask_ratio = 0.15
            random_mask = (torch.rand(x.shape, device=device) < mask_ratio) & (~padding_mask.unsqueeze(-1))
            
            # Clone input data and corrupt the targeted values with zeros
            inputs_masked = x.clone()
            inputs_masked[random_mask] = 0.0
            
            # Forward pass through the foundation model
            optimizer.zero_grad()
            predictions = model(inputs_masked, padding_mask=padding_mask)
            
            # Calculate mean squared error only on the masked/hidden continuous elements
            loss = criterion(predictions, x)
            masked_loss = loss[random_mask].mean() if random_mask.sum() > 0 else loss.mean()
            
            # Backpropagation step
            masked_loss.backward()
            optimizer.step()
            
            epoch_loss += masked_loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Masked Reconstruction MSE Loss: {epoch_loss/len(dataloader):.4f}")
        
    return model.state_dict()

if __name__ == "__main__":
    # Test pretraining on hospital 1 data
    _ = pretrain_local_hospital('data/hospital_1_data.csv', epochs=2)
