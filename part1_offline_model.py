import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import json
import os

def main():
    # Load data
    df = pd.read_parquet('output/wesad_eda_features_phase3.parquet')
    
    # Drop NaNs
    df = df.dropna()
    
    # Identify features and metadata
    metadata_cols = ['subject', 'window_id', 'binary_label']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    subjects = df['subject'].unique()
    print(f"Subjects found: {subjects}")
    
    # We will save the model for a specific target subject to simulate the edge device later.
    # Let's pick the last subject in the list as the edge subject.
    target_edge_subject = subjects[-1]
    print(f"\nTarget edge subject for later phases: {target_edge_subject}")
    
    results = {}
    
    for test_subject in subjects:
        # Split data
        train_df = df[df['subject'] != test_subject]
        test_df = df[df['subject'] == test_subject]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['binary_label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['binary_label'].values
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Logistic Regression
        # max_iter increased to ensure convergence, C=1.0 is default
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[test_subject] = {
            'accuracy': acc,
            'cm': cm
        }
        
        print(f"\nSubject {test_subject} (Held-out)")
        print(f"Accuracy: {acc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        # Save the model if it's our target edge subject
        if test_subject == target_edge_subject:
            os.makedirs('output/models', exist_ok=True)
            
            # Save weights, bias, scaler mean/std
            weights = clf.coef_[0].tolist()
            bias = float(clf.intercept_[0])
            mean = scaler.mean_.tolist()
            std = scaler.scale_.tolist()
            
            model_data = {
                'target_subject': target_edge_subject,
                'weights': weights,
                'bias': bias,
                'scaler_mean': mean,
                'scaler_std': std,
                'feature_names': feature_cols
            }
            
            with open('output/models/generic_model_data.json', 'w') as f:
                json.dump(model_data, f, indent=4)
                
            print(f"\nSaved generic model for target subject {target_edge_subject} to 'output/models/generic_model_data.json'")
            
            # Save the test data for this subject so we can use it in Part 2
            test_df.to_parquet(f'output/models/test_data_{target_edge_subject}.parquet', index=False)

if __name__ == "__main__":
    main()
