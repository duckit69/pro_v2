import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # clip to prevent overflow

def main():
    # Load data
    df = pd.read_parquet('output/wesad_eda_features_phase3.parquet')
    df = df.dropna()
    
    metadata_cols = ['subject', 'window_id', 'binary_label']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    subjects = df['subject'].unique()
    
    # Store metrics for average improvement calculation
    generic_accs = []
    generic_f1s = []
    inc_accs = []
    inc_f1s = []
    
    for test_subject in subjects:
        # Split data
        train_df = df[df['subject'] != test_subject]
        test_df = df[df['subject'] == test_subject].copy()
        
        # Chronological sort
        test_df = test_df.sort_values('window_id')
        
        X_train = train_df[feature_cols].values
        y_train = train_df['binary_label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['binary_label'].values
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Generic Logistic Regression
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y_train)
        
        # Get starting weights and bias
        generic_weights = clf.coef_[0].copy()
        generic_bias = float(clf.intercept_[0])
        
        inc_weights = generic_weights.copy()
        inc_bias = generic_bias
        
        initial_lr = 0.001
        decay_rate = 0.01
        
        # Replay buffer setup
        buffer_size = 10
        replay_buffer = [] # list of (features, label) tuples
        buffer_idx = 0
        updates_count = 0
        
        generic_preds = []
        inc_preds = []
        current_threshold = 0.5
        
        # Simulation Loop
        for i in range(len(X_test_scaled)):
            features = X_test_scaled[i]
            label = y_test[i]
            
            # 1. Predict with Generic (fixed) Model
            g_z = np.dot(generic_weights, features) + generic_bias
            g_pred_prob = sigmoid(g_z)
            generic_preds.append(1 if g_pred_prob >= 0.5 else 0)
            
            # 2. Predict with Incremental Model
            i_z = np.dot(inc_weights, features) + inc_bias
            i_pred_prob = sigmoid(i_z)
            inc_preds.append(1 if i_pred_prob >= current_threshold else 0)
            
            # Dynamic LR
            current_lr = initial_lr / (1 + decay_rate * updates_count)

            # 3. SGD Update Incremental Model
            error = label - i_pred_prob
            inc_weights += current_lr * error * features
            inc_bias += current_lr * error
            updates_count += 1
            
            # 4. Replay buffer logic
            if len(replay_buffer) < buffer_size:
                replay_buffer.append((features, label))
            else:
                replay_buffer[buffer_idx] = (features, label)
            buffer_idx = (buffer_idx + 1) % buffer_size
            
            if updates_count % 5 == 0:
                # Replay buffer pass
                for b_features, b_label in replay_buffer:
                    b_z = np.dot(inc_weights, b_features) + inc_bias
                    b_pred_prob = sigmoid(b_z)
                    b_error = b_label - b_pred_prob
                    inc_weights += current_lr * b_error * b_features
                    inc_bias += current_lr * b_error
                    
                # Evaluate Dynamic Threshold
                best_f1 = -1
                best_thresh = 0.5
                for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    temp_preds = []
                    temp_labels = []
                    for b_features, b_label in replay_buffer:
                        b_z = np.dot(inc_weights, b_features) + inc_bias
                        b_pred_prob = sigmoid(b_z)
                        temp_preds.append(1 if b_pred_prob >= thresh else 0)
                        temp_labels.append(b_label)
                    f1 = f1_score(temp_labels, temp_preds, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh
                current_threshold = best_thresh
                    
        # Metrics for the subject
        g_acc = accuracy_score(y_test, generic_preds)
        g_f1 = f1_score(y_test, generic_preds, zero_division=0)
        g_cm = confusion_matrix(y_test, generic_preds)
        
        i_acc = accuracy_score(y_test, inc_preds)
        i_f1 = f1_score(y_test, inc_preds, zero_division=0)
        i_cm = confusion_matrix(y_test, inc_preds)
        
        generic_accs.append(g_acc)
        generic_f1s.append(g_f1)
        inc_accs.append(i_acc)
        inc_f1s.append(i_f1)
        
        print(f"\n--- Subject {test_subject} ---")
        print(f"Generic Model     - Acc: {g_acc:.4f}, F1: {g_f1:.4f}")
        print(f"Generic CM:\n{g_cm}")
        print(f"Incremental Model - Acc: {i_acc:.4f}, F1: {i_f1:.4f}")
        print(f"Incremental CM:\n{i_cm}")
        print(f"Improvement       - Acc: {i_acc - g_acc:+.4f}, F1: {i_f1 - g_f1:+.4f}")
        
    print("\n=============================================")
    print("FINAL AVERAGE RESULTS ACROSS ALL SUBJECTS:")
    avg_g_acc = np.mean(generic_accs)
    avg_g_f1 = np.mean(generic_f1s)
    avg_i_acc = np.mean(inc_accs)
    avg_i_f1 = np.mean(inc_f1s)
    
    print(f"Generic     -> Avg Acc: {avg_g_acc:.4f}, Avg F1: {avg_g_f1:.4f}")
    print(f"Incremental -> Avg Acc: {avg_i_acc:.4f}, Avg F1: {avg_i_f1:.4f}")
    print(f"Improvement -> Acc: {avg_i_acc - avg_g_acc:+.4f}, F1: {avg_i_f1 - avg_g_f1:+.4f}")

if __name__ == "__main__":
    main()
