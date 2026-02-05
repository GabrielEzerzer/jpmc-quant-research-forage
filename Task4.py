import pandas as pd
import numpy as np

# Constants
B = 10                  
MIN_BUCKET_SIZE = 200   


n_prefix = None
k_prefix = None

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def build_prefixes():
    #Load Data and group by FICO score
    global n_prefix, k_prefix
    
    data_path = r"File Path"
    df = pd.read_csv(data_path, usecols=["fico_score", "default"])
    df["default"] = df["default"].astype(int)

    grouped = (df.groupby("fico_score")["default"]
                 .agg(n="size", k="sum")
                 .reset_index()
                 .sort_values("fico_score")
                 .reset_index(drop=True))

    # Convert to numpy for performance
    n_vals = grouped["n"].to_numpy(dtype=int)
    k_vals = grouped["k"].to_numpy(dtype=int)

    # Prefix sums
    n_prefix = np.zeros(len(n_vals) + 1, dtype=int)
    k_prefix = np.zeros(len(k_vals) + 1, dtype=int)
    n_prefix[1:] = np.cumsum(n_vals)
    k_prefix[1:] = np.cumsum(k_vals)
    
    scores = grouped["fico_score"].to_numpy(dtype=int)
    return scores

def range_sum(a, b):
    #Calculates n and k for the range
    n_ab = n_prefix[b+1] - n_prefix[a]
    k_ab = k_prefix[b+1] - k_prefix[a]
    return n_ab, k_ab

def score(a, b):
    #Calculate log-likelihood for the range
    n_ab, k_ab = range_sum(a, b)
    
    # Enforce minimum bucket size constraint
    if n_ab < MIN_BUCKET_SIZE:
        return -np.inf

    # Maximum Likelihood Estimate for probability of default
    eps = 1e-15
    p = k_ab / n_ab
    p = np.clip(p, eps, 1 - eps)
    
    
    return k_ab * np.log(p) + (n_ab - k_ab) * np.log(1 - p)

def calculate_iv(a, b, total_k, total_non_k):
    
    n_ab, k_ab = range_sum(a, b)
    non_k_ab = n_ab - k_ab
    
    dist_k = k_ab / total_k if total_k > 0 else 0
    dist_non_k = non_k_ab / total_non_k if total_non_k > 0 else 0
    
    if dist_k == 0 or dist_non_k == 0:
        return 0
        
    woe = np.log(dist_non_k / dist_k)
    return (dist_non_k - dist_k) * woe

def dp_optimal_buckets(scores):
    #Finds the globally optimal split points using Dynamic Programming.
    M = len(scores)

    dp = np.full((B + 1, M), -np.inf, dtype=float)
    prev = np.full((B + 1, M), -1, dtype=int)

    # Base case: 1 bucket
    for b in range(M):
        s = score(0, b)
        if s != -np.inf:
            dp[1][b] = s
            prev[1][b] = 0

    # DP transition
    for j in range(2, B + 1):
        for b in range(M):
            # Pruning
            for a in range(1, b + 1):
                if dp[j-1][a-1] == -np.inf:
                    continue

                s = score(a, b)
                if s == -np.inf:
                    continue

                total_score = dp[j-1][a-1] + s
                if total_score > dp[j][b]:
                    dp[j][b] = total_score
                    prev[j][b] = a

    return dp, prev

def create_rating_map(boundaries):
    
    thresholds = [b[1] for b in boundaries]
    
    def rating_fn(fico):
        if fico < 300: return 1
        if fico > 850: return 10
        
        for i, threshold in enumerate(thresholds):
            if fico <= threshold:
                return i + 1
        return len(thresholds)
        
    return rating_fn

def main():
    print("Pre-processing unique FICO scores and cumulative counts...")
    scores = build_prefixes()
    M = len(scores)
    
    total_k = k_prefix[M]
    total_non_k = n_prefix[M] - k_prefix[M]
    
    print(f"Running Global Optimization (DP) for {B} buckets across {M} score points...")
    dp, prev = dp_optimal_buckets(scores)
    
    # Check if a valid partition exists
    if dp[B][M-1] == -np.inf:
        print("Error: No optimal 10-bucket mix found. Try reducing MIN_BUCKET_SIZE.")
        return

    # Backtracking to recover optimal boundaries
    b = M - 1
    bucket_indices = []
    for j in range(B, 0, -1):
        a = prev[j][b]
        bucket_indices.append((a, b))
        b = a - 1
    
    bucket_indices = bucket_indices[::-1] # Convert to Cronological order

    # Report results
    print("\n" + "="*80)
    print(f"{'BUCKET':<7} | {'FICO RANGE':<12} | {'SAMPLES':<8} | {'DEFAULTS':<8} | {'DEF %':<8} | {'IV':<8}")
    print("-" * 80)
    
    final_boundaries = []
    total_iv = 0
    for i, (a, b) in enumerate(bucket_indices):
        min_f, max_f = scores[a], scores[b]
        final_boundaries.append((min_f, max_f))
        
        n_ab, k_ab = range_sum(a, b)
        iv_ab = calculate_iv(a, b, total_k, total_non_k)
        total_iv += iv_ab
        
        print(f"{i+1:<7} | {min_f:>3}-{max_f:<8} | {n_ab:<8} | {k_ab:<8} | {k_ab/n_ab:>7.2%} | {iv_ab:.4f}")

    print("-" * 80)
    print(f"{'TOTAL':<7} | {' '*12} | {n_prefix[M]:<8} | {k_prefix[M]:<8} | {k_prefix[M]/n_prefix[M]:>7.2%} | {total_iv:.4f}")
    print("="*80)
    
    # Final Model Export
    rating_fn = create_rating_map(final_boundaries)
    print(f"Rating Map generated. Final Information Value (IV): {total_iv:.4f}")
    
    # Quick sanity check
    print("\nValidation:")
    print(f"FICO 400 -> Rating {rating_fn(400)}")
    print(f"FICO 850 -> Rating {rating_fn(850)}")

if __name__ == "__main__":
    main()