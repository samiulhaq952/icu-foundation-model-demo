import numpy as np
import pandas as pd
import os

def generate_icu_hospital_data(hospital_id, num_patients=100, max_timesteps=48):
    """
    Generates synthetic, irregular time-series ICU data for a specific hospital.
    Features simulate continuous vitals and lab results.
    """
    np.random.seed(hospital_id)
    all_records = []
    features = ['heart_rate', 'spo2', 'systolic_bp', 'temperature', 'creatinine']
    
    for patient_id in range(num_patients):
        seq_len = np.random.randint(12, max_timesteps) 
        base_values = {
            'heart_rate': np.random.uniform(60, 100),
            'spo2': np.random.uniform(94, 100),
            'systolic_bp': np.random.uniform(100, 140),
            'temperature': np.random.uniform(36.5, 37.5),
            'creatinine': np.random.uniform(0.6, 1.2)
        }
        
        for t in range(seq_len):
            record = {
                'hospital_id': f"HOSP_{hospital_id}",
                'patient_id': f"H{hospital_id}_P{patient_id}",
                'timestep': t,
            }
            for feat in features:
                drift = np.random.normal(0, 0.05 * base_values[feat])
                record[feat] = base_values[feat] + drift
            all_records.append(record)
            
    df = pd.DataFrame(all_records)
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    for i in range(1, 4):
        df_hospital = generate_icu_hospital_data(hospital_id=i, num_patients=150)
        df_hospital.to_csv(f'data/hospital_{i}_data.csv', index=False)
        print(f"Generated data for Hospital {i}: {df_hospital.shape} rows saved.")
