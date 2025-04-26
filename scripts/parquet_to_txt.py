import pandas as pd

# Load the parquet file
df = pd.read_parquet('../data/posts-with-tickers.parquet')
# Save the DataFrame to a text file
for index, row in df.iterrows():
    ticker = row['ticker']
    date = row['created_utc']
    with open('../data/updated_tickers.txt', 'a') as f:
        f.write(f"{ticker} / {date} \n")