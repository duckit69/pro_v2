import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

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
        
        generic_weights = clf.coef_[0].copy()
        generic_bias = float(clf.intercept_[0])
        
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
        inc_preds = []
        
        for i in range(len(X_test_scaled)):
            features = X_test_scaled[i]
            label = y_test[i]
            
            i_z = np.dot(inc_weights, features) + inc_bias
            i_pred_prob = sigmoid(i_z)
            inc_preds.append(1 if i_pred_prob >= 0.5 else 0)
            
            for _ in range(10):
                i_z_loop = np.dot(inc_weights, features) + inc_bias
                i_pred_prob_loop = sigmoid(i_z_loop)
                error = label - i_pred_prob_loop
                clipped_error = np.clip(error, -1.0, 1.0)
                ewc_penalty_weights = ewc_lambda * F_weights * (inc_weights - generic_weights)
                ewc_penalty_bias = ewc_lambda * F_bias * (inc_bias - generic_bias)
                grad_w = clipped_error * features - ewc_penalty_weights
                grad_b = clipped_error - ewc_penalty_bias
                grad_w = np.clip(grad_w, -1.0, 1.0)
                grad_b = np.clip(grad_b, -1.0, 1.0)
                inc_weights += lr * grad_w
                inc_bias += lr * grad_b
                
            updates_count += 1
            
            if len(replay_buffer) < buffer_size:
                replay_buffer.append((features, label))
            else:
                replay_buffer[buffer_idx] = (features, label)
            buffer_idx = (buffer_idx + 1) % buffer_size
            
            if updates_count % 2 == 0:
                for b_features, b_label in replay_buffer:
                    b_z = np.dot(inc_weights, b_features) + inc_bias
                    b_pred_prob = sigmoid(b_z)
                    b_error = b_label - b_pred_prob
                    b_clipped_error = np.clip(b_error, -1.0, 1.0)
                    b_ewc_penalty_weights = ewc_lambda * F_weights * (inc_weights - generic_weights)
                    b_ewc_penalty_bias = ewc_lambda * F_bias * (inc_bias - generic_bias)
                    b_grad_w = b_clipped_error * b_features - b_ewc_penalty_weights
                    b_grad_b = b_clipped_error - b_ewc_penalty_bias
                    b_grad_w = np.clip(b_grad_w, -1.0, 1.0)
                    b_grad_b = np.clip(b_grad_b, -1.0, 1.0)
                    inc_weights += lr * b_grad_w
                    inc_bias += lr * b_grad_b
                    
            if updates_count % 1000 == 0:
                for _ in range(50):
                    for b_features, b_label in replay_buffer:
                        b_z = np.dot(inc_weights, b_features) + inc_bias
                        b_pred_prob = sigmoid(b_z)
                        b_error = b_label - b_pred_prob
                        b_clipped_error = np.clip(b_error, -1.0, 1.0)
                        b_ewc_penalty_weights = ewc_lambda * F_weights * (inc_weights - generic_weights)
                        b_ewc_penalty_bias = ewc_lambda * F_bias * (inc_bias - generic_bias)
                        b_grad_w = b_clipped_error * b_features - b_ewc_penalty_weights
                        b_grad_b = b_clipped_error - b_ewc_penalty_bias
                        b_grad_w = np.clip(b_grad_w, -1.0, 1.0)
                        b_grad_b = np.clip(b_grad_b, -1.0, 1.0)
                        inc_weights += lr * b_grad_w
                        inc_bias += lr * b_grad_b
                        
        i_acc = accuracy_score(y_test, inc_preds)
        i_f1 = f1_score(y_test, inc_preds, zero_division=0)
        cm = confusion_matrix(y_test, inc_preds)
        
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
            
        print(f"| {test_subject} | {i_acc:.4f} | {i_f1:.4f} | {cm_str} |")

if __name__ == '__main__':
    main()
