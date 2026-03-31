import pandas as pd
import yfinance as yf
import glob
import os
import sys
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

picks_dir = "/Users/onkarj012/Projects/major_pro/intraday_antigravity/recommendations"
pick_files = glob.glob(os.path.join(picks_dir, "picks_*.csv"))

all_picks = []
for f in pick_files:
    try:
        df = pd.read_csv(f)
        if 'picks_for_date' in df.columns:
            all_picks.append(df)
    except pd.errors.EmptyDataError:
        pass

if not all_picks:
    print("No picks found.")
    sys.exit(0)

combined_picks = pd.concat(all_picks, ignore_index=True)
required_cols = ['picks_for_date', 'stock', 'direction', 'entry_price', 'target_price', 'stop_loss']
combined_picks = combined_picks.dropna(subset=required_cols)

# Deduplicate in case same date and same stock appear in multiple pick files
combined_picks = combined_picks.drop_duplicates(subset=['picks_for_date', 'stock'])

results = []

for date, group in combined_picks.groupby('picks_for_date'):
    print(f"Processing date: {date}")
    start_date = date
    # yfinance end date is exclusive, so date + 1
    end_date = (pd.to_datetime(date) + timedelta(days=1)).strftime('%Y-%m-%d')
    date_dt = pd.to_datetime(date).date()
    today = datetime.now().date()
    
    # We cannot test future predictions yet
    if date_dt >= today:
        print(f"  Skipping future/current date {date} as data may be incomplete")
        continue

    # Get all tickers at once for speed, although yf.download handles lists
    stocks_list = [row['stock'] + ".NS" for _, row in group.iterrows()]
    tickers = " ".join(stocks_list)
    print(f"  Fetching 1m data for {len(stocks_list)} stocks...")
    
    # download using interval
    try:
        df_group = yf.download(tickers, start=start_date, end=end_date, interval='1m', progress=False)
        if df_group.empty:
            df_group = yf.download(tickers, start=start_date, end=end_date, interval='5m', progress=False)
    except Exception as e:
        print(f"  Error fetching bulk data: {e}")
        continue
        
    for _, row in group.iterrows():
        symbol = row['stock'] + ".NS"
        direction = row['direction']
        entry = row['entry_price']
        target = row['target_price']
        sl = row['stop_loss']
        
        hit_target = False
        hit_sl = False

        if df_group.empty:
            print(f"    No data found for {symbol} on {date}")
            continue
            
        # extract symbol data
        if len(stocks_list) > 1:
            try:
                # Handle multi-index columns
                high_col = df_group['High'][symbol] if 'High' in df_group.columns and symbol in df_group['High'].columns else None
                low_col = df_group['Low'][symbol] if 'Low' in df_group.columns and symbol in df_group['Low'].columns else None
            except KeyError:
                continue
        else:
            high_col = df_group['High'] if 'High' in df_group.columns else None
            low_col = df_group['Low'] if 'Low' in df_group.columns else None
        
        if high_col is None or low_col is None or high_col.dropna().empty:
            print(f"    Missing intraday data for {symbol}")
            continue

        # create a temp dataframe
        temp_df = pd.DataFrame({'High': high_col, 'Low': low_col}).dropna()

        for index, ticker_row in temp_df.iterrows():
            high = ticker_row['High'].item() if hasattr(ticker_row['High'], 'item') else ticker_row['High']
            low = ticker_row['Low'].item() if hasattr(ticker_row['Low'], 'item') else ticker_row['Low']
            
            if direction == 'LONG':
                if low <= sl:
                    hit_sl = True
                if high >= target:
                    hit_target = True
                    
                if hit_sl and not hit_target:
                    results.append({'date': date, 'stock': symbol, 'direction': direction, 'outcome': 'Loss'})
                    break
                elif hit_target and not hit_sl:
                    results.append({'date': date, 'stock': symbol, 'direction': direction, 'outcome': 'Win'})
                    break
                elif hit_target and hit_sl:
                    results.append({'date': date, 'stock': symbol, 'direction': direction, 'outcome': 'Loss (Simul)'})
                    break
                    
            elif direction == 'SHORT':
                if high >= sl:
                    hit_sl = True
                if low <= target:
                    hit_target = True
                    
                if hit_sl and not hit_target:
                    results.append({'date': date, 'stock': symbol, 'direction': direction, 'outcome': 'Loss'})
                    break
                elif hit_target and not hit_sl:
                    results.append({'date': date, 'stock': symbol, 'direction': direction, 'outcome': 'Win'})
                    break
                elif hit_target and hit_sl:
                    results.append({'date': date, 'stock': symbol, 'direction': direction, 'outcome': 'Loss (Simul)'})
                    break
        
        # If we loop through the whole day and neither hit
        if not hit_target and not hit_sl:
            results.append({'date': date, 'stock': symbol, 'direction': direction, 'outcome': 'Timeout'})

res_df = pd.DataFrame(results)
if not res_df.empty:
    print("\nResults summary:")
    print(res_df['outcome'].value_counts())
    
    wins = res_df['outcome'].str.contains('Win').sum()
    total_trades = len(res_df)
    win_rate = (wins / total_trades) * 100
    print(f"\nTotal Trades Evaluated: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    res_df.to_csv('recommendations_accuracy.csv', index=False)
    print("Saved raw results to 'recommendations_accuracy.csv'")
else:
    print("No results could be calculated.")
