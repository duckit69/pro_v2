import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def compute_rolling_accuracy(y_true, y_pred, window=100):
    rolling_acc = []
    for i in range(len(y_true)):
        start_idx = max(0, i - window + 1)
        window_true = y_true[start_idx:i+1]
        window_pred = y_pred[start_idx:i+1]
        acc = np.mean(np.array(window_true) == np.array(window_pred))
        rolling_acc.append(acc)
    return rolling_acc

def main():
    print("Loading data...")
    df = pd.read_parquet('output/wesad_eda_features_phase3.parquet')
    df = df.dropna()
    
    metadata_cols = ['subject', 'window_id', 'binary_label']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    subjects = sorted(df['subject'].unique())
    all_results = {}
    
    for test_subject in subjects:
        print(f"Simulating incremental learning for {test_subject}...")
        
        # 1. Loads the feature matrix and labels (chronological order).
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
        
        # 2. Initialises the generic logistic regression model.
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y_train)
        
        generic_weights = clf.coef_[0].copy()
        generic_bias = float(clf.intercept_[0])
        
        # EWC Prep
        p_train = clf.predict_proba(X_train_scaled)[:, 1]
        gradient_weights = (y_train - p_train)[:, np.newaxis] * X_train_scaled
        F_weights = np.mean(gradient_weights**2, axis=0)
        F_bias = np.mean((y_train - p_train)**2)
        ewc_lambda = 0.001
        
        inc_weights = generic_weights.copy()
        inc_bias = generic_bias
        lr = 0.1
        
        buffer_size = 100
        replay_buffer = []
        buffer_idx = 0
        updates_count = 0
        
        y_pred_list = []
        y_true_list = []
        
        # Data structures to store cumulative metrics
        cumulative_cm_list = []
        
        # 3. Simulates the incremental updates.
        for i in range(len(X_test_scaled)):
            features = X_test_scaled[i]
            label = y_test[i]
            
            # 4.1 & 4.2 Record true and predicted (before update)
            i_z = np.dot(inc_weights, features) + inc_bias
            pred = 1 if sigmoid(i_z) >= 0.5 else 0
            y_pred_list.append(pred)
            y_true_list.append(int(label))
            
            # 5. Record cumulative confusion matrix after each window
            cm = confusion_matrix(y_true_list, y_pred_list, labels=[0, 1])
            cumulative_cm_list.append(cm.tolist())
            
            # Aggressive tuning updates
            for _ in range(10):
                i_z_loop = np.dot(inc_weights, features) + inc_bias
                error = label - sigmoid(i_z_loop)
                clipped_error = np.clip(error, -1.0, 1.0)
                
                ewc_penalty_weights = ewc_lambda * F_weights * (inc_weights - generic_weights)
                ewc_penalty_bias = ewc_lambda * F_bias * (inc_bias - generic_bias)
                
                grad_w = np.clip(clipped_error * features - ewc_penalty_weights, -1.0, 1.0)
                grad_b = np.clip(clipped_error - ewc_penalty_bias, -1.0, 1.0)
                
                inc_weights += lr * grad_w
                inc_bias += lr * grad_b
                
            updates_count += 1
            
            # Replay buffer handling
            if len(replay_buffer) < buffer_size:
                replay_buffer.append((features, label))
            else:
                replay_buffer[buffer_idx] = (features, label)
            buffer_idx = (buffer_idx + 1) % buffer_size
            
            if updates_count % 2 == 0:
                for b_features, b_label in replay_buffer:
                    b_z = np.dot(inc_weights, b_features) + inc_bias
                    b_error = b_label - sigmoid(b_z)
                    b_clipped_error = np.clip(b_error, -1.0, 1.0)
                    b_ewc_penalty_weights = ewc_lambda * F_weights * (inc_weights - generic_weights)
                    b_ewc_penalty_bias = ewc_lambda * F_bias * (inc_bias - generic_bias)
                    b_grad_w = np.clip(b_clipped_error * b_features - b_ewc_penalty_weights, -1.0, 1.0)
                    b_grad_b = np.clip(b_clipped_error - b_ewc_penalty_bias, -1.0, 1.0)
                    inc_weights += lr * b_grad_w
                    inc_bias += lr * b_grad_b
                    
        # 4.3 Compute rolling accuracy over the last L = 100 predictions
        roll_inc = compute_rolling_accuracy(y_true_list, y_pred_list, window=100)
        
        # 6. Save for each subject
        all_results[test_subject] = {
            'y_true': y_true_list,
            'y_pred': y_pred_list,
            'rolling_accuracy_100': roll_inc,
            'cumulative_confusion_matrix': cumulative_cm_list
        }
    
    # 6.1 Save to disk
    output_path = Path('output/simulation_rolling_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f)
        
    print(f"\nSuccessfully simulated 15 subjects and saved results to {output_path}")

if __name__ == '__main__':
    main()
