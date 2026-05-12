import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def simulate_incremental_ablation(X, y, generic_model, lr, k_updates, buffer_size, replay_every, clip):
    w = generic_model.coef_[0].copy()
    b = generic_model.intercept_[0].copy()
    replay_buffer = []
    y_pred = []
    
    for i, (x, y_true) in enumerate(zip(X, y)):
        # predict
        p = 1/(1+np.exp(-(np.dot(w, x)+b)))
        pred = 1 if p>=0.5 else 0
        y_pred.append(pred)
        
        # update on true label
        for _ in range(k_updates):
            p_loop = 1/(1+np.exp(-(np.dot(w, x)+b)))
            error = y_true - p_loop
            grad_w = lr * error * x
            grad_b = lr * error
            if clip and np.linalg.norm(grad_w) > 1.0:
                grad_w = grad_w / np.linalg.norm(grad_w)
            w += grad_w
            b += grad_b
            
        # add to replay buffer
        if buffer_size > 0:
            replay_buffer.append((x, y_true))
            if len(replay_buffer) > buffer_size:
                replay_buffer.pop(0)
                
        # replay
        if buffer_size > 0 and replay_every is not None and (i+1) % replay_every == 0:
            for x_rep, y_rep in replay_buffer:
                p_rep = 1/(1+np.exp(-(np.dot(w, x_rep)+b)))
                error_rep = y_rep - p_rep
                grad_w_rep = lr * error_rep * x_rep
                grad_b_rep = lr * error_rep
                if clip and np.linalg.norm(grad_w_rep) > 1.0:
                    grad_w_rep = grad_w_rep / np.linalg.norm(grad_w_rep)
                w += grad_w_rep
                b += grad_b_rep
                
    return np.array(y_pred)

def main():
    print("Loading data...")
    df = pd.read_parquet('output/wesad_eda_features_phase3.parquet')
    df = df.dropna()
    
    metadata_cols = ['subject', 'window_id', 'binary_label']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    target_subjects = ['S4', 'S8', 'S9','S10', 'S17', 'S14']
    
    configs = [
        {"name": "Full aggressive (baseline)", "lr": 0.1, "k": 10, "buf": 50, "rep": 2, "clip": True},
        {"name": "No replay",                  "lr": 0.1, "k": 10, "buf": 0,  "rep": None, "clip": True},
        {"name": "Low LR (0.01)",              "lr": 0.01,"k": 1,  "buf": 50, "rep": 2, "clip": True},
        {"name": "No clipping",                "lr": 0.1, "k": 10, "buf": 50, "rep": 2, "clip": False},
        {"name": "No multiple updates",        "lr": 0.1, "k": 1,  "buf": 50, "rep": 2, "clip": True}
    ]
    
    results = {cfg['name']: {} for cfg in configs}
    
    for test_subject in target_subjects:
        print(f"Processing {test_subject}...")
        train_df = df[df['subject'] != test_subject]
        test_df = df[df['subject'] == test_subject].copy()
        test_df = test_df.sort_values('window_id')
        
        X_train = train_df[feature_cols].values
        y_train = train_df['binary_label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['binary_label'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y_train)
        
        for cfg in configs:
            y_pred = simulate_incremental_ablation(
                X_test_scaled, y_test, clf, 
                lr=cfg['lr'], k_updates=cfg['k'], 
                buffer_size=cfg['buf'], replay_every=cfg['rep'], clip=cfg['clip']
            )
            f1 = f1_score(y_test, y_pred, zero_division=0)
            results[cfg['name']][test_subject] = f1
            
    print("\n\nAblation Study Results:")
    print("| Configuration | S4 (high generic) | S8 (low generic) | S9 (low generic) | S10 (low generic) | S14 (low generic) |S17 (low generic) | Average |")
    for cfg in configs:
        name = cfg['name']
        s4 = results[name]['S4']
        s8 = results[name]['S8']
        s9 = results[name]['S9']
        s10 = results[name]['S10']
        s14 = results[name]['S14']
        s17 = results[name]['S17']
        avg = (s4 + s8 + s9 + s10 + s14 + s17) / 6
        print(f"| {name} | {s4:.4f} | {s8:.4f} | {s9:.4f} | {s10:.4f} | {s14:.4f} | {s17:.4f} | {avg:.4f} |")

if __name__ == '__main__':
    main()
