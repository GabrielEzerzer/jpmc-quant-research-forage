import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def build_features(df):
    """
    Calculates features and returns the feature matrix X and target y.
    """
    eps = 1e-9
    df["debt_to_income"] = df["total_debt_outstanding"] / (df["income"] + eps)
    df["loan_to_income"] = df["loan_amt_outstanding"] / (df["income"] + eps)

    # Features used for PD model: FICO, DTI, LTI, Years Employed, Credit Lines
    feature_cols = ["fico_score", "debt_to_income", "loan_to_income", "years_employed", "credit_lines_outstanding"]
    X = df[feature_cols].to_numpy(dtype=float)
    
    # We keep loan_amt_outstanding separately for Expected Loss calculations
    loan_amt = df["loan_amt_outstanding"].to_numpy(dtype=float)
    y = df["default"].to_numpy(dtype=int)
    
    return X, y, loan_amt

def calculate_log_loss(y, pd_predicted):
    """
    Calculates the binary cross-entropy (log-loss).
    """
    eps = 1e-15 # Avoid log(0)
    pd_predicted = np.clip(pd_predicted, eps, 1 - eps)
    return -np.mean(y * np.log(pd_predicted) + (1 - y) * np.log(1 - pd_predicted))

def compute_pd(X, beta):
    """
    Computes the Probability of Default using the logistic function.
    """
    score = np.dot(X, beta)
    # Robust sigmoid to avoid overflow
    return np.where(score >= 0, 
                    1 / (1 + np.exp(-score)), 
                    np.exp(score) / (1 + np.exp(score)))

def train_step(X, y, beta, learning_rate):
    """
    Performs one step of gradient descent for logistic regression.
    """
    PD = compute_pd(X, beta)
    error = y - PD
    gradient = np.dot(X.T, error) / len(y)
    beta = beta + learning_rate * gradient
    return beta

if __name__ == "__main__":
    # 1. Load Data
    data_path = r"File Path"
    df = load_data(data_path)
    X_full, y_full, loan_amt_full = build_features(df)
    
    # 2. Stratified Shuffle Split
    # Goal: 7000 train, 3000 test, maintaining default rate
    np.random.seed(42) # For repeatability
    
    idx_default = np.where(y_full == 1)[0]
    idx_no_default = np.where(y_full == 0)[0]
    
    np.random.shuffle(idx_default)
    np.random.shuffle(idx_no_default)
    
    # Using 70% as split ratio
    split_ratio = 0.7
    n_default_train = int(len(idx_default) * split_ratio)
    n_no_default_train = int(len(idx_no_default) * split_ratio)
    
    train_idx = np.concatenate([idx_default[:n_default_train], idx_no_default[:n_no_default_train]])
    test_idx = np.concatenate([idx_default[n_default_train:], idx_no_default[n_no_default_train:]])
    
    # Shuffle the final train/test indices so they aren't ordered by class
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    X_train_raw = X_full[train_idx]
    y_train = y_full[train_idx]
    loan_train = loan_amt_full[train_idx]
    
    X_test_raw = X_full[test_idx]
    y_test = y_full[test_idx]
    loan_test = loan_amt_full[test_idx]
    
    print(f"Dataset Split:")
    print(f"  Train: {len(y_train)} rows (Default Rate: {np.mean(y_train):.2%})")
    print(f"  Test:  {len(y_test)} rows (Default Rate: {np.mean(y_test):.2%})")
    
    # 3. Scale Features using Train Mean/Std
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)
    std[std == 0] = 1.0 # Avoid division by zero
    
    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std
    
    # Add Intercept Column (column of ones) after scaling
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    
    # 4. Initialize Parameters (beta vector)
    beta = np.zeros(X_train.shape[1])
    learning_rate = 0.1
    
    try:
        num_epochs_input = input("\nEnter number of epochs (*recommended 500*): ")
        num_epochs = int(num_epochs_input) if num_epochs_input else 500
    except EOFError:
        num_epochs = 500
    
    print(f"\nTraining Model...")
    header = f"{'Epoch':<8} | {'Train Loss':<12} | {'Test Loss':<12} | {'EL Train':<12} | {'EL Test':<12}"
    print(header)
    print("-" * len(header))
    
    for epoch in range(num_epochs):
        # Training Step
        beta = train_step(X_train, y_train, beta, learning_rate)
        
        # Evaluation
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            pd_train = compute_pd(X_train, beta)
            pd_test = compute_pd(X_test, beta)
            
            train_loss = calculate_log_loss(y_train, pd_train)
            test_loss = calculate_log_loss(y_test, pd_test)
            
            # Expected Loss = PD * LGD * EAD (Assuming LGD is 0.9)
            el_train = np.mean(pd_train * 0.9 * loan_train)
            el_test = np.mean(pd_test * 0.9 * loan_test)
            
            print(f"{epoch:<8} | {train_loss:<12.6f} | {test_loss:<12.6f} | {el_train:<12.2f} | {el_test:<12.2f}")

    print(f"\nFinal Test Log-Loss: {calculate_log_loss(y_test, compute_pd(X_test, beta)):.6f}")
    print(f"Final Average PD (Test): {np.mean(compute_pd(X_test, beta)):.2%}")
