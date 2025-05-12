import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Function to generate random date ranges between 2002 and 2025
def random_dates(start_year=2002, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    random_start_date = start + timedelta(days=random.randint(0, (end - start).days))
    random_end_date = random_start_date + timedelta(days=random.randint(30, 365))  # Random period of 1-2 years
    return random_start_date.strftime('%Y-%m-%d'), random_end_date.strftime('%Y-%m-%d'), random_start_date, random_end_date

df = {}
capital = 100000
position = 0
entry_price = 0
equity_curve = []
dates = []
returns = []


summary = {
    "positive":[],
    "negative":[]
    }
return_map = {

}

def call():
    global capital
    global position
    global entry_price
    global equity_curve
    global dates
    global returns 
    global summary
    global return_map
    # Investment amount for each strategy
    investment_amount = 100000  # ₹1,00,000 per investment

    # Generate random start and end dates
    start_date, end_date, random_start_date,random_end_date = random_dates()
    # # Calculate the duration between start and end date
    # duration_in_days = (end_date - start_date).days
    # duration_in_months = relativedelta(end_date, start_date).months + relativedelta(end_date, start_date).years * 12
    if(str(start_date)+str(end_date) in df):
        return 
    df[str(start_date)+str(end_date)] = True
    
    today = datetime.today()
    if(random_end_date > today):
        return "NA"
    print("------------------")
    x = random_end_date-random_start_date
    duration = x.days
    
    # print(f"Random Date Range: {start_date} to {end_date}")
    try:

        # Step 1: Download data
        nifty = yf.download("TATAMOTORS.NS", start=start_date, end=end_date)
        nifty.dropna(inplace=True)

        # Step 2: Calculate returns
        nifty['returns'] = nifty['Close'].pct_change()
        nifty['up'] = (nifty['returns'] > 0).astype(int)

        # Step 3: Binomial CDF
        window = 20
        n = window
        p = 0.5
        nifty['successes'] = nifty['up'].rolling(window=window).sum()
        nifty['binom_cdf'] = nifty['successes'].apply(lambda x: binom.cdf(x, n, p) if pd.notnull(x) else np.nan)

        # Step 4: EMAs and signals
        nifty['fast_ema'] = nifty['binom_cdf'].ewm(span=21, adjust=False).mean()
        nifty['slow_ema'] = nifty['binom_cdf'].ewm(span=484, adjust=False).mean()
        nifty['long_signal'] = (nifty['fast_ema'] > nifty['slow_ema']) & (nifty['fast_ema'].shift(1) <= nifty['slow_ema'].shift(1))
        nifty['exit_signal'] = (nifty['fast_ema'] < nifty['slow_ema']) & (nifty['fast_ema'].shift(1) >= nifty['slow_ema'].shift(1))


        list_from_df = nifty.values.tolist()

        # Step 5: Backtest loop
        capital = 100000
        position = 0
        entry_price = 0
        equity_curve = []
        dates = []
        returns = []
        
        for i in range(1, len(list_from_df)-1):
            row = list_from_df[i]
            next_row = list_from_df[i+1]

            if row[11] and position == 0:
                entry_price = next_row[3]
                position = capital / entry_price

            # Exit
            elif row[12] and position > 0:
                exit_price = next_row[3]
                pnl = (exit_price - entry_price) * position
                capital += pnl
                returns.append(pnl / (entry_price * position))
                position = 0

            # Mark equity
            mark_price = row[0]
            total = capital if position == 0 else position * mark_price
            equity_curve.append(total)
            dates.append(row[0])

        # Final value if still in position
        if position > 0:
            final_price = list_from_df[-1][0]
            capital = position * final_price
            equity_curve.append(capital)
            dates.append(nifty.index[-1])

        # Step 6: Results
        equity_series = pd.Series(equity_curve, index=dates)
        returns_series = pd.Series(returns)
        total_return = (equity_curve[-1] - 100000) / 100000
        num_trades = len(returns_series)
        win_rate = returns_series.sum() / num_trades if num_trades else 0
        max_drawdown = (equity_series / equity_series.cummax() - 1).min()
        if(total_return < 0):
            summary["negative"].append(total_return)
        else:
            summary["positive"].append(total_return)
        
        if(duration not in return_map):
            return_map[duration] = []
        
        return_map[duration].append([start_date,end_date, total_return])
        # Step 7: Print + Plot
        print(f"Total Return: {total_return}")
        print(f"Number of Trades: {num_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Max Drawdown: {max_drawdown:.2%}")

        # plt.figure(figsize=(14,6))
        # equity_series.plot()
        # plt.title("Equity Curve - Binomial EMA Strategy")
        # plt.grid(True)
        # plt.show()
        
        print("------------------")
    except:
        print("dsfg")
        return "NA"

for i in range(1000):
    call()


print(len(summary["positive"]), max(summary["positive"]),len(summary["negative"]), min(summary["negative"]))
print(return_map)