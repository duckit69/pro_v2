import pandas as pd
import numpy as np
import serial
import time
import json
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser(description="Send WESAD data to ESP32 for incremental learning")
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--subject', type=str, default='S9', help='Subject to simulate')
    parser.add_argument('--model-json', type=str, default='output/generic_model.json',
                        help='Path to generic_model.json exported by export_generic_model.py')
    args = parser.parse_args()

    print(f"Loading data for subject {args.subject}...")
    
    # Load dataset
    df = pd.read_parquet('output/wesad_eda_features_phase3.parquet')
    df = df.dropna()
    
    test_df = df[df['subject'] == args.subject].copy()
    if len(test_df) == 0:
        print(f"Error: Subject {args.subject} not found in dataset.")
        return
        
    test_df = test_df.sort_values('window_id')
    
    metadata_cols = ['subject', 'window_id', 'binary_label']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    print(f"Identified {len(feature_cols)} features.")
    
    X_test = test_df[feature_cols].values
    y_test = test_df['binary_label'].values
    
    # Load scaler parameters from generic_model.json (produced by export_generic_model.py).
    # This guarantees the Python-side normalisation is IDENTICAL to what is
    # hardcoded in model_weights.h on the device.
    model_json_path = Path(args.model_json)
    if not model_json_path.exists():
        print(f"WARNING: {model_json_path} not found. Falling back to refitting scaler from training data.")
        train_df = df[df['subject'] != args.subject]
        X_train = train_df[feature_cols].values
        scaler = StandardScaler()
        scaler.fit(X_train)
    else:
        with open(model_json_path) as f:
            model_data = json.load(f)
        scaler = StandardScaler()
        scaler.mean_  = np.array(model_data['scaler_mean'])
        scaler.scale_ = np.array(model_data['scaler_std'])
        scaler.var_   = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler.mean_)
        print(f"Scaler loaded from {model_json_path} (held-out: {model_data.get('held_out_subject','?')})")

    X_test_scaled = scaler.transform(X_test)
    
    print(f"Connecting to ESP32 on {args.port} at {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=5)
        # Allow ESP32 to reset
        time.sleep(2)
        # Clear any initial garbage from serial buffer
        ser.reset_input_buffer()
        print("Connected.")
    except Exception as e:
        print(f"Failed to connect to Serial Port: {e}")
        print("Continuing in dry-run mode (printing to console instead).")
        ser = None
        
    for i in range(len(X_test_scaled)):
        features = X_test_scaled[i]
        label = y_test[i]
        
        # 1. Send features
        # Note: Using len(feature_cols) which is 15.
        feature_str = ",".join([f"{f:.4f}" for f in features]) + "\n"
        
        if ser:
            ser.write(feature_str.encode('utf-8'))
            ser.flush()
            
            # 2. Wait for ESP32 inference result
            response = ser.readline().decode('utf-8').strip()
            print(f"Window {i+1}/{len(X_test_scaled)} - Sent features. ESP32 inference: {response}")
            
            # 3. Send true label
            label_str = f"label:{label}\n"
            ser.write(label_str.encode('utf-8'))
            ser.flush()
            
            # Wait for ESP32 to acknowledge the update (optional, but good for sync)
            ack = ser.readline().decode('utf-8').strip()
            print(f"  Sent true label: {label}. ESP32 update ack: {ack}")
        else:
            print(f"DRY RUN Window {i+1}/{len(X_test_scaled)}: Sent features -> {feature_str.strip()}")
            print(f"DRY RUN: Sent label -> label:{label}")
            
        time.sleep(0.01) # small delay to not overwhelm the serial buffer
        
    print("\nData transmission complete. Sending END_SHIFT...")
    
    if ser:
        ser.write(b"END_SHIFT\n")
        ser.flush()
        
        print("Waiting for final accuracy and confusion matrix from ESP32...")
        # Read lines until we hit a timeout or a specific end marker
        while True:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                break # Timeout
            print(f"ESP32: {line}")
            
        ser.close()
    else:
        print("DRY RUN: Sent END_SHIFT")

if __name__ == "__main__":
    main()
