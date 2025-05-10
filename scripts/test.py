import pandas as pd

# Load the parquet file
df = pd.read_parquet('../data/posts-with-tickers.parquet')
for index, row in df.iterrows():
    print(row)