import torch
import torch.nn as nn
import os

class ICUTransformerFM(nn.Module):
    def __init__(self, num_features=5, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, max_len=48):
        super(ICUTransformerFM, self).__init__()
        
        # 1. Continuous Input Embedding Projector
        self.input_projection = nn.Linear(num_features, d_model)
        
        # 2. Learnable Positional Encodings for temporal order
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        # 3. Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Head (Maps back to feature dimension for self-supervised reconstruction)
        self.output_projection = nn.Linear(d_model, num_features)
        
    def forward(self, x, padding_mask=None):
        # x shape: [Batch, Timesteps, Features]
        
        # Project continuous features to d_model hidden dimensions
        x = self.input_projection(x) # Shape: [Batch, Timesteps, d_model]
        
        # Add temporal position identifiers
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Pass through transformer layers using the key padding mask
        # Note: PyTorch's src_key_padding_mask expects True for positions that should be ignored
        out = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Map back to original physiological feature spaces
        predictions = self.output_projection(out) # Shape: [Batch, Timesteps, Features]
        
        return predictions

if __name__ == "__main__":
    # Create models folder directory structure
    os.makedirs('models', exist_ok=True)
    
    # Quick architecture and shape validation check
    model = ICUTransformerFM()
    
    # Mock data resembling our dataset batch shape
    mock_x = torch.randn(4, 48, 5)
    mock_mask = torch.zeros(4, 48, dtype=torch.bool)
    mock_mask[:, 40:] = True # simulate padding last 8 timesteps
    
    output = model(mock_x, padding_mask=mock_mask)
    print("--- Model Structural Test Successful ---")
    print("Input Tensor Shape:", mock_x.shape)
    print("Output Foundation Model Prediction Shape:", output.shape)
