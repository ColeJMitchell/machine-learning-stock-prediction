import ollama
import pandas as pd

# Load the parquet file
df = pd.read_parquet('../data/posts-with-tickers.parquet')
tickers = []

# Iterate through the posts and classify their corresponding tickers
for post in df['selftext']:
    response = ollama.chat(
        model='llama3:70b',  
        messages=[
        {'role': 'user', 
        'content': f'what is the stock ticker for this text "{post}" you have to answer in one word with no period. If you cant determine it just ourput None. Also if there is no text provided output None.'}
        ]
    )
    if len(response['message']['content']) > 5:
        response['message']['content'] = 'None'
    tickers.append(response['message']['content'])

df['ticker'] = tickers

# Save the updated DataFrame to the parquet file
df.to_parquet('data/posts-with-tickers.parquet')
