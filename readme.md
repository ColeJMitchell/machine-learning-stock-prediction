# Machine Learning Stock Prediction
- Author(s): Cole Mitchel, Ava Ginsberg, and Benjamin Gregory
- Date: 2025-05-02

## Overview
_

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
| `scripts/ticker_classifier.py` | | |
| `scripts/parquet_to_txt.py` | | |
| `scripts/manual_verification.py` | | |
| `data_collection.ipynb` | Collect and generate data from r/wallstreetbets for further analysis and use | |
| `general_analysis.ipynb` | Simple exploration of data collected from r/wallstreetbets | |
| `model_evaluation.ipynb` | Evaluate the model trained in `stock_predictor.ipynb` with test stocks and simulation | |
| `sentiment_analysis.ipynb` | Provide sentiment analysis scores for data collected from r/wallstreetbets | |
| `stock_predictor.ipynb` | Create initial stock prediction model through time series data | |
| `ticker_attribution.ipynb` | Exploration of ticker attribution for data from r/wallstreetbets | |
| `readme.md` | Markdown file for project overview | |
| `requirements.txt`| Python package requirements and installation | Python3.12.* |

### Data Sources and Structure
Whether the raw data sources are included in the public repository or are generated throughout the usage, notice that there are numerous data files included in the `./data/` directory. These files are generated as checkpoints, throughout the modeling process. Description of the data files, origins, and usage are listed below. 

| Filename (`./data/<filename>`) | Source/Input Reference (`./<filename>`) | Usage | Desc |
|---|---|---|---|
| `kaggle-reddit-wsb.parquet` | `data_collection.ipynb` |  |  |
| `merged_reddit_wsb.parquet` | `data_collection.ipynb` |  | This is an unused artifact from mismatched naming. |
| `merged-reddit-wsb.parquet` | `data_collection.ipynb` |  |  |
| `nasdaq_tickers.parquet` | External |  |  |
| `reddit_wsb.csv` | `data_collection.ipynb` |  |  |
| `sp500_companies.csv` | External |  |  |
| `posts-with-tickers.parquet` | External |  |
| `wallstreetbets-comment-collection-wss.parquet` | `sentiment_analysis.ipynb` |  |  |
| `wallstreetbets-comment-collection.parquet` | `data_collection.ipynb` |  |  |
| `wallstreetbets-collection-wss.parquet` | `sentiment_analysis.ipynb` |  |  |
| `wallstreetbets-collection.parquet` | `data_collection.ipynb` |  |  |
| `kaggle-reddit-wsb.parquet` | `data_collection.ipynb` |  |  |
| `reddit_wsb.txt` | `data_collection.ipynb` |  | WSB Data Dump from Kaggle Source |
| `test_tickers.txt` | External |  |  |
| `tickers.txt` | External |  |  |
| `updated_tickers.txt` | `scripts/parquet_to_txt.py` |  |  |
| `test_stocks/<Stock Ticker>.csv` | External | `./model_evaluation.ipynb` | Recent stock data used for model test cases. Downloaded from NASDAQ Site. |
| `stock_prediction.h5` | Model saved in google colab from `./stock_predictor.ipynb` |  |  |

## Developement Structure
Within this project, data aquisition was aimed at collecting submissions and comments from r/wallstreet bets. With this data, external LLMs among other techniques are used to attemp attributing some public stock ticker or industry code for every collected data element. Next, we additionally employ sentiment analysis over the elements of this collected corpus. With sentiments, dates, and reference to financial elements, we can employ a strategy to reinforce pretrained financial models with publicly available sentiment.

https://github.com/user-attachments/assets/c20b8bdf-d8b8-4360-8f3b-f3d961f56d32

https://github.com/user-attachments/assets/94f5e936-03b9-479e-b18d-574bdf93e96d

### Initial Analysis
Some very simple initial analysis is used to provide a basic understanding of the aquired data prior to moving forward with this project. This analysis show that while data collection was performed effectively, the attribution of industry or financial identifiers was essentially impossible given the very limited resources and time frame of this project. That is about 4 weeks during the spring of 2025.


### Model Design
TODO: Fill in by Cole Mitchel.
### Model Training
TODO: Fill in by Cole Mitchel.

## Outcomes
### Model Analysis
### Final Review

## Further Exploration
### Unanswered Questions
### Shortcomings and resolution
### Possibilty of Pipeline Developement
