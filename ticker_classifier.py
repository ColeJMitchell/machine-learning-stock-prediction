import ollama
import pandas as pd

# Load the parquet file
df = pd.read_parquet('data/wallstreetbets-collection.parquet')

# Iterate through the posts in the parquet file and 
with open('data/tickers.txt', 'w') as f:
    for post in df['selftext']:
        response = ollama.chat(
            model='llama3',  
            messages=[
            {'role': 'user', 
            'content': f'what is the stock ticker for this text "{post}" you have to answer in one word with no period. If you cant determine it just ourput None. Also if there is no text provided output None.'}
        ]
        )
        if len(response['message']['content']) > 5:
            response['message']['content'] = 'None'
        f.write(response['message']['content'] + '\n')

