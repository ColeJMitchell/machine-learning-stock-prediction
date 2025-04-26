import pandas as pd


# Manual verification of the LLM's output
df = pd.read_parquet('../data/posts-with-tickers.parquet')

# Iterate through the posts in the parquet file and print the post and its corresponding ticker
for index, row in df.iterrows():
    post = row['selftext']
    ticker = row['ticker']
    print(f"Post: {post}\nTicker: {ticker}\n")