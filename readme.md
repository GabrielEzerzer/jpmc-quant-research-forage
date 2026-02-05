# JPMC Quantitative Research Job Simulation Files
Approach to each task can be found in their respective python files.

## Project Summary
### Task 1: Natural Gas Price Modelling

Modelled monthly natural gas prices using a trend + seasonality framework
Generated price estimates for arbitrary dates and extrapolated one year forward

### Task 2: Gas Storage Contract Pricing

Built a cash-flow-based pricing model for a gas storage contract
Incorporated injection/withdrawal schedules and operational constraints

### Task 3: Probability of Default (PD) Modelling

Implemented logistic regression from scratch (no sklearn)
Engineered credit-relevant features (DTI, LTI, employment history)
Evaluated using log-loss, calibration, and expected loss metrics

### Task 4: FICO Score Quantization

Converted continuous FICO scores into 10 categorical risk ratings
Used dynamic programming to maximize within-bucket likelihood
Enforced minimum bucket size for statistical stability
Evaluated bucket quality using Information Value (IV)



## Results Summary
### Probability of Default Model

Out-of-sample average predicted PD closely matched observed default rate
Stable convergence under gradient descent
Expected Loss computed using PD × LGD × Exposure

### FICO Quantization (10 Buckets)

Monotonic relationship between FICO score and default rate
High-risk borrowers isolated into low-score buckets
Total Information Value (IV): ~0.80, indicating strong predictive power
Final output: interpretable FICO → risk rating map suitable for categorical ML models
