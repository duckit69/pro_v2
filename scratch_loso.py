import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def main():
    df = pd.read_parquet('output/wesad_eda_features_phase3.parquet')
    df = df.dropna()
    
    metadata_cols = ['subject', 'window_id', 'binary_label']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    subjects = sorted(df['subject'].unique())
    
    print("| Subject | Accuracy | F1-Score | Confusion Matrix [TN, FP; FN, TP] |")
    print("|---------|----------|----------|-----------------------------------|")
    
    for test_subject in subjects:
        train_df = df[df['subject'] != test_subject]
        test_df = df[df['subject'] == test_subject]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['binary_label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['binary_label'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y_train)
        
        y_pred = clf.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            cm_str = f"[{tn}, {fp}; {fn}, {tp}]"
        elif cm.shape == (1, 1):
            if y_test[0] == 0:
                cm_str = f"[{cm[0,0]}, 0; 0, 0]"
            else:
                cm_str = f"[0, 0; 0, {cm[0,0]}]"
        else:
            cm_str = str(cm.tolist())
            
        print(f"| {test_subject} | {acc:.4f} | {f1:.4f} | {cm_str} |")

if __name__ == '__main__':
    main()
