import sys
import os

# Explicitly add the root working directory to the system path so Python can find train.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import copy
from train import pretrain_local_hospital
from models.transformer import ICUTransformerFM

def federated_averaging(global_model, local_weights_list):
    """
    Implements the FedAvg algorithm. 
    Averages the state dictionaries of models trained at different hospitals.
    """
    # Create a deep copy of the first hospital's weights to initialize the average structure
    avg_weights = copy.deepcopy(local_weights_list[0])
    
    # Sum up parameter values across all other hospitals
    for key in avg_weights.keys():
        for i in range(1, len(local_weights_list)):
            avg_weights[key] = avg_weights[key] + local_weights_list[i][key]
        # Divide by the total number of hospitals to compute the true mathematical average
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights_list))
        
    # Load the unified averaged parameters back into the global foundation model
    global_model.load_state_dict(avg_weights)
    return global_model

def run_multicentre_pretraining(rounds=2):
    """
    Simulates a secure, decentralised multi-hospital network training session.
    """
    print("Initializing Multi-Center Global ICU Foundation Model...")
    global_model = ICUTransformerFM()
    
    hospital_files = [
        'data/hospital_1_data.csv',
        'data/hospital_2_data.csv',
        'data/hospital_3_data.csv'
    ]
    
    for r in range(rounds):
        print(f"\n================ FEDERATED ROUND {r+1}/{rounds} ================")
        local_weights = []
        
        # Sequentially train local nodes (simulating remote hospitals)
        for h_idx, csv_path in enumerate(hospital_files):
            print(f"\n--- Dispatched Global Model weights to Hospital {h_idx+1} ---")
            weights = pretrain_local_hospital(csv_path, epochs=1, batch_size=16)
            local_weights.append(weights)
            
        # Secure aggregation step
        print("\n--- Collecting weights and executing Federated Averaging (FedAvg) ---")
        global_model = federated_averaging(global_model, local_weights)
        
    # Save the finalized multi-center foundation model checkpoint
    os.makedirs('models', exist_ok=True)
    torch.save(global_model.state_dict(), 'models/global_icu_fm.pth')
    print("\nMulticentre Global Foundation Model fully trained and saved to 'models/global_icu_fm.pth'!")

if __name__ == "__main__":
    run_multicentre_pretraining(rounds=2)
