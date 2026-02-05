
import matplotlib.pyplot as plt
import csv
from datetime import datetime, date
import numpy as np

# Path to the dataset
Dataset = "File Path"

def read_data(file_path):
    dates = []
    prices = []
    with open(file_path, "r") as file:
        csvFile = csv.reader(file)
        next(csvFile) 
        for line in csvFile:
            d = datetime.strptime(line[0], "%m/%d/%y").date()
            p = float(line[1])
            dates.append(d)
            prices.append(p)
    return dates, prices

def fit_trend(prices):
    x = np.arange(len(prices))
    y = np.array(prices)
    

    slope, intercept = np.polyfit(x, y, 1)
    

    trend = slope * x + intercept
    

    residuals = y - trend
    
    return trend, residuals, slope, intercept

def analyze_seasonality(dates, residuals):

    monthly_residuals = {i: [] for i in range(1, 13)}
    
    for d, res in zip(dates, residuals):
        month = d.month
        monthly_residuals[month].append(res)
        

    seasonal_indices = {}
    for month in range(1, 13):
        if monthly_residuals[month]:
            seasonal_indices[month] = np.mean(monthly_residuals[month])
        else:
            seasonal_indices[month] = 0 
            
    return seasonal_indices

def plot_analysis(dates, prices, trend, residuals, seasonal_indices):


    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(dates, prices, label="Actual Prices")
    plt.plot(dates, trend, label="Linear Trend", linestyle="--", color='red')
    plt.title("Natural Gas Prices and Linear Trend")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    

    plt.subplot(2, 1, 2)
    plt.plot(dates, residuals, label="Residuals (Seasonality + Noise)", marker='o', alpha=0.5)
    

    seasonal_signal = [seasonal_indices[d.month] for d in dates]
    plt.plot(dates, seasonal_signal, label="Average Seasonal Pattern", color='green', linewidth=2)
    
    plt.title("Seasonality: Residuals and Average Monthly Effect")
    plt.xlabel("Date")
    plt.ylabel("Deviations from Trend")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def predict_price(dates, query_date, slope, intercept, seasonal_indices):

    if isinstance(query_date, str):
        try:
            query_date = datetime.strptime(query_date, "%m/%d/%y").date()
        except ValueError:
            print("Error: Please use the format MM/DD/YY (e.g., 10/31/24).")
            return None


    if query_date <= dates[0]:
        x_new = (query_date - dates[0]).days / 30.44
    elif query_date <= dates[-1]:
        i = 0
        for idx in range(len(dates) - 1):
            if dates[idx] <= query_date <= dates[idx+1]:
                i = idx
                break
        

        denom = (dates[i+1] - dates[i]).days
        w = (query_date - dates[i]).days / denom if denom > 0 else 0
        x_new = i + w
    else:
        x_new = (len(dates) - 1) + (query_date - dates[-1]).days / 30.44


    predicted_trend = slope * x_new + intercept
    
 
    predicted_seasonality = seasonal_indices[query_date.month]
    predicted_price = predicted_trend + predicted_seasonality
    
    return predicted_price


def price_contract(dates, actions, max_storage, slope, intercept, seasonal_indices, 
                   inj_rate_daily, wd_rate_daily, storage_cost_daily):
    """
    Calculates the value of a storage contract with rate and capacity constraints.
    - Accrues storage costs daily based on inventory.
    - Clamps volumes to feasible amounts (rate limits & capacity validation).
    
    Args:
        dates: list of training dates (for prediction context)
        actions: list of (type, date_str, volume) tuples. type='injection' or 'withdrawal'
        max_storage: maximum inventory capacity units
        slope, intercept, seasonal_indices: pricing model parameters
        inj_rate_daily: max injection per day
        wd_rate_daily: max withdrawal per day
        storage_cost_daily: cost per unit per day to hold gas
    """
    
    # Initialize Portfolio State
    inventory = 0.0
    cash_flow = 0.0
    
    # Convert action dates to objects and sort
    # SORTING RULE: Process Withdrawals BEFORE Injections on the same day (common convention)
    # Mapping type to priority: withdrawal=0, injection=1
    parsed_actions = []
    for typ, date_str, vol in actions:
        d = datetime.strptime(date_str, "%m/%d/%y").date()
        parsed_actions.append((typ, d, vol))
        
    parsed_actions.sort(key=lambda x: (x[1], 0 if x[0] == 'withdrawal' else 1))
    
    if not parsed_actions:
        return 0.0
        
    # Start simulation from the first action date
    prev_date = parsed_actions[0][1]
    
    print("\n--- Contract Simulation ---")
    
    for typ, curr_date, vol_requested in parsed_actions:
        # 1. Accrue Storage Costs for the period [prev_date, curr_date)
        days_elapsed = (curr_date - prev_date).days
        if days_elapsed < 0:
            print(f"Error: Dates out of order {prev_date} -> {curr_date}")
            return None
            
        period_cost = inventory * storage_cost_daily * days_elapsed
        cash_flow -= period_cost
        
        # 2. Calculate Constraints for this specific action
        # Max volume we can physically move since last event based on rates
        if typ == 'injection':
            max_rate_vol = inj_rate_daily * days_elapsed if days_elapsed > 0 else float('inf') 
            # (If same day, assume we have full daily capacity available or handle differently. 
            #  Here assuming full daily rate is available per step if days=0 is edge case, 
            #  but usually steps are apart. Let's start with daily limitation logic.)
            if days_elapsed == 0: max_rate_vol = inj_rate_daily # Allow 1 day's worth on same day? Or 0? Let's be generous.
            
            # Constraint: Cannot exceed max storage
            space_available = max_storage - inventory
            feasible_vol = min(vol_requested, max_rate_vol, space_available)
            
            # Execute Injection
            price = predict_price(dates, curr_date, slope, intercept, seasonal_indices)
            cost_to_buy = feasible_vol * price
            cash_flow -= cost_to_buy
            inventory += feasible_vol
            
            print(f"Inject {feasible_vol:.0f} (Req: {vol_requested:.0f}) @ {price:.2f} on {curr_date}. Inventory: {inventory:.0f}")
            
        elif typ == 'withdrawal':
            max_rate_vol = wd_rate_daily * days_elapsed if days_elapsed > 0 else float('inf')
            if days_elapsed == 0: max_rate_vol = wd_rate_daily
            
            # Constraint: Cannot withdraw more than we have
            gas_available = inventory
            feasible_vol = min(vol_requested, max_rate_vol, gas_available)
            
            # Execute Withdrawal
            price = predict_price(dates, curr_date, slope, intercept, seasonal_indices)
            revenue_from_sell = feasible_vol * price
            cash_flow += revenue_from_sell
            inventory -= feasible_vol
            
            print(f"Withdraw {feasible_vol:.0f} (Req: {vol_requested:.0f}) @ {price:.2f} on {curr_date}. Inventory: {inventory:.0f}")
            
        else:
            print(f"Unknown action type: {typ}")
            
        # Update pointer
        prev_date = curr_date
        
    # Check final state (Optional: Penalize leftover inventory or force sell)
    # For now, just report value.
    print(f"End Simulation. Final Inventory: {inventory:.0f}")
    return cash_flow

    
if __name__ == "__main__":
    # 1. Load Data
    dates, prices = read_data(Dataset)
    print(f"Loaded {len(dates)} data points.")

    # 2. Trend Analysis
    trend, residuals, slope, intercept = fit_trend(prices)
    #print(f"Trend Slope: {slope:.4f} per month")
    
    # 3. Seasonality Analysis
    seasonal_indices = analyze_seasonality(dates, residuals)
    #print("Seasonal Indices (Month 1-12):")
    #for m in range(1, 13):
        #print(f"Month {m}: {seasonal_indices[m]:.4f}")
        
    # 4. Plot
    #plot_analysis(dates, prices, trend, residuals, seasonal_indices)

    # 5. Predict Prices
    query_date_str = input("Enter the date for which you want to predict the price (MM/DD/YY): ")
    try:
        query_date_obj = datetime.strptime(query_date_str, "%m/%d/%y").date()
        price_pred = predict_price(dates, query_date_obj, slope, intercept, seasonal_indices)
        print(f"\nPrediction for {query_date_obj}: {price_pred:.2f}")
    except:
        print("Skipping prediction due to invalid date.")

    # 6. Price a contract
    # Define contract parameters
    inj_rate = 50000  # Max injection rate per day
    wd_rate = 50000   # Max withdrawal rate per day
    max_storage = 100000 # Max total storage
    storage_cost = 0.001 # Cost per unit per day (approx 10 cents per 100 units/day)
    
    # Define a list of actions (Type, Date, Volume)
    # Strategy: Buy in Summer (Jun/Jul), Sell in Winter (Dec/Jan)
    contract_actions = [
        ('injection', '06/15/24', 20000),  # Summer injection
        ('injection', '07/15/24', 40000),  # Summer injection
        ('withdrawal', '12/15/24', 10000), # Winter withdrawal
        ('withdrawal', '01/15/25', 40000)  # Peak Winter withdrawal
    ]
    
    value = price_contract(dates, contract_actions, max_storage, slope, intercept, seasonal_indices,
                           inj_rate, wd_rate, storage_cost)
                           
    print(f"\nTotal Contract Value: ${value:,.2f}")
