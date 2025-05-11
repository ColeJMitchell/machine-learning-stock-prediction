# Machine Learning Stock Prediction
- Author(s): Cole Mitchell, Ava Ginsberg, and Benjamin Gregory
- Date: 2025-05-02

## Overview
This project aims to explore multiple machine learning techniques, frameworks, and algorithms introduced in the curriculum for Computer Science 424: Introduction to Machine Learning at Lafayette College. In this repository, machine learning strategies are employed while modeling the closing costs of public stocks for on US market. After developing a LSTM-based architecture, we attempted to prove that machine learning techniques can outperform random investment in stocks. We also tested an open source natural language tool kit model to determine if it would yield more accurate predictions than the LSTM and random strategies.

## Organization
### Usage and Setup
The enclosed material includes `./requirements.txt`, containing the relevant packages and versions for the enclosed code. It is recommend that you open a virtual python environment and install packages within, avoiding possible conflicts with local data. The program installs several materials locally, including but not limited to the NLTK (Natural Language ToolKit) corpus data. For these local installs, we open directories at the project root; however, these directories are not pushed to the public branch. With this in mind, run any provided code with caution as to not overwrite local data. Additionally, the source code leverages the public reddit data api. The user must include a `.env` file at the root of this project with the following contents,
```
CLIENT_SECRET= <provided by reddit>
CLIENT_ID= <provided by reddit>
USERNAME= <provided by user for reddit>
PASSWORD= <proided by user for reddit>
KAGGLE_USERNAME = <Kaggle API username>
KAGGLE_KEY = <Kaggle API key>
```
These environmental attributes are retrieved prior to collecting reddit submission and comment data. Note that reddit makes no guarantee regarding consistent rate limits for the public API. In some cases, reddit API access may not be available due to circumstances on their end. Aditionally, the collected reddit data is stored in the `./data/` directory, but may not be available in the public branch for this project. Where the usage of the provided notebooks is concerned, different environemts have been used for the varying objectives of thus project. In many cases, a venv with Python3.12.* was used after downloading required packages via the `./requirements.txt` file. As the developement and exploration of this project has updated over time and will continue to evolve, the reproducabilty of the provided notebooks may be effected by these changes. In most cases, model traing was run through google colab (~2025-05-09) and all other program code was run in the virtual python environment referenced previously via Windows 11 and WSL2 (Ubuntu).

### Notebook and Python Script Structure.

| Filename (`./<filename>`) | Purpose | Details |
|---|---|---|
| `scripts/ticker_classifier.py` |  | In raw text or csv format. |
| `scripts/parquet_to_txt.py` |  |  |
| `scripts/manual_verification.py` |  |  |
| `data_collection.ipynb` | Collect and generate data from r/wallstreetbets for further analysis and use |  |
| `general_analysis.ipynb` | Simple exploration of data collected from r/wallstreetbets |  |
| `model_evaluation.ipynb` | Evaluate the model trained in `stock_predictor.ipynb` with test stocks and simulation |  |
| `sentiment_analysis.ipynb` | Provide sentiment analysis scores for data collected from r/wallstreetbets |  |
| `stock_predictor.ipynb` | Create initial stock prediction model through time series data |  |
| `ticker_attribution.ipynb` | Exploration of ticker attribution for data from r/wallstreetbets |  |
| `readme.md` | Markdown file for project overview |  |
| `requirements.txt`| Python package requirements and installation | Python3.12.* |

### Data Sources and Structure
Whether the raw data sources are included in the public repository or are generated throughout the usage, notice that there are numerous data files included in the `./data/` directory. These files are generated as checkpoints, throughout the modeling process. Description of the data files, origins, and usage are listed below. 

| Filename (`./data/<filename>`) | Source/Reference (`./<filename>`) | Details |
|---|---|---|
| `kaggle-reddit-wsb.parquet` | `data_collection.ipynb` |  |
| `merged_reddit_wsb.parquet` | `data_collection.ipynb` | This is an unused artifact from mismatched naming. |
| `merged-reddit-wsb.parquet` | `data_collection.ipynb` |  |
| `nasdaq_tickers.parquet` | External |  | 
| `reddit_wsb.csv` | `data_collection.ipynb` |  | 
| `sp500_companies.csv` | External |  |
| `posts-with-tickers.parquet` | External |  |
| `wallstreetbets-comment-collection-wss.parquet` | `sentiment_analysis.ipynb` |  | 
| `wallstreetbets-comment-collection.parquet` | `data_collection.ipynb` |  | 
| `wallstreetbets-collection-wss.parquet` | `sentiment_analysis.ipynb` |  | 
| `wallstreetbets-collection.parquet` | `data_collection.ipynb` |  | 
| `kaggle-reddit-wsb.parquet` | `data_collection.ipynb` |  | 
| `reddit_wsb.txt` | `data_collection.ipynb` | WSB Data Dump from Kaggle Source |
| `test_tickers.txt` | External |  | 
| `tickers.txt` | External |  | 
| `updated_tickers.txt` | `scripts/parquet_to_txt.py` |  | 
| `test_stocks/<Stock Ticker>.csv` | External | Recent stock data used for model test cases. Downloaded from NASDAQ Site. |
| `stock_prediction.h5` | Model saved in google colab from `./stock_predictor.ipynb` |  |

## Developement Structure
### Data Aquisition
Within this project, data aquisition was aimed at collecting submissions and comments from r/wallstreet bets. While data collection was performed effectively, the attribution of industry or financial identifiers was extremely challenging given the very limited resources and time frame of this project. The data collected spans 4 weeks during the spring of 2025. With this in mind, supplementary data was downloaded from kaggle. The final collection of reddit submission data ranged from 2020 to 2025 with around 50000 elements. With this data, the open source Llama large language model was used to successfully attribute 67 unique stock ticker to the posts. Next, we additionally employed sentiment analysis over the elements of this collected corpus. With sentiments, dates, and reference to financial elements, we devised a strategy to use pretrained sentiment analysis models with the reddit data.

### Overview of data collection: [link](https://github.com/ColeJMitchell/machine-learning-stock-prediction/blob/main/media/DataCollectionNotebookOverview.mp4):
https://github.com/user-attachments/assets/804e38a0-81fd-4d67-a802-059461bee602

### Overview of sentiment analysis: [link](https://github.com/ColeJMitchell/machine-learning-stock-prediction/blob/main/media/SentimentAnalysisNotebookOverview.mp4):
https://github.com/user-attachments/assets/47c3e3bf-1c93-4f0e-8f75-b10799cd7f80

### Stock data collection and preprocessing: [link](https://github.com/ColeJMitchell/machine-learning-stock-prediction/blob/main/media/data_collection.mp4):
https://github.com/user-attachments/assets/584fa63e-a2d8-42e2-9a4d-6be22e602dee

### Data split and model architecture: [link](https://github.com/ColeJMitchell/machine-learning-stock-prediction/blob/main/media/data_sets_and_model_architecture.mp4):
https://github.com/user-attachments/assets/825df98c-babc-4417-8dca-37b59570d723

### Model training and testing: [link](https://github.com/ColeJMitchell/machine-learning-stock-prediction/blob/main/media/training_and_testing.mp4):
https://github.com/user-attachments/assets/edbdf418-3c7e-4596-9c6c-7fee7128137c
