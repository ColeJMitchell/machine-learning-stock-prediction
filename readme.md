# Machine Learning Stock Prediction
- Author(s): Cole Mitchell, Ava Ginsberg, and Benjamin Gregory
- Date: 2025-05-02

### Executive Report Link: [link](https://github.com/user-attachments/files/20213875/Executive.Report.pdf)
### Model Training And Initial Testing Link: 
<a href="https://colab.research.google.com/github/ColeJMitchell/machine-learning-stock-prediction/blob/main/model_notebooks/stock_predictor.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
### Model Evaluation Link: 
<a href="https://colab.research.google.com/github/ColeJMitchell/machine-learning-stock-prediction/blob/main/model_notebooks/model_evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
### Reddit Data Collection Notebook Link: [link](https://github.com/ColeJMitchell/machine-learning-stock-prediction/blob/8550a789dda467557738e0fce97293f2236bdfec/data_collection_notebooks/data_collection.ipynb)
### Sentiment Analysis Notebook Link: [link](https://github.com/ColeJMitchell/machine-learning-stock-prediction/blob/8550a789dda467557738e0fce97293f2236bdfec/data_collection_notebooks/sentiment_analysis.ipynb)

## Overview
This project aims to explore multiple machine learning techniques, frameworks, and algorithms introduced in the curriculum for Computer Science 424: Introduction to Machine Learning at Lafayette College. In this repository, machine learning strategies are employed while modeling the closing costs of public stocks for on US market. After developing a LSTM-based architecture, we attempted to prove that machine learning techniques can outperform random investment in stocks. We also tested several open source large language models to see if they could correctly identify stock tickers based on text data.

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

## Data Aquisition
Within this project, data aquisition was aimed at collecting submissions and comments from r/wallstreet bets. While data collection was performed effectively, the attribution of industry or financial identifiers was extremely challenging given the very limited resources and time frame of this project. The data collected spans 4 weeks during the spring of 2025. With this in mind, supplementary data was downloaded from kaggle. The final collection of reddit submission data ranged from 2020 to 2025 with around 50000 elements. With this data, the open source llama3 large language model was used to successfully attribute 67 unique stock ticker to the posts. The python script for classifying the tickers can be found in scripts/ticker_classifier.py. Next, we additionally employed sentiment analysis over the elements of this collected corpus. With sentiments, dates, and reference to financial elements, we devised a strategy to use pretrained sentiment analysis models with the reddit data.

## LSTM Model Overview And Architecture
We chose the long short-term memory architecture over other machine learning architectures because of its ability to to capture and retain long-term dependencies in sequential data. This model was well-suited to our problem, as the previous month's stock prices provide strong predictive signals for the stock's behavior on the target day. We decided to also add several dense layers after the LSTM layers in an attempt to have our model learn additional non-linear features in the stock data. We had one output neuron at the end which returned an estimate for the normalized stock price. It was also feasible to take a binary classification approach, where the output would simply be a prediction of whether the price will go up or down. This was never integrated because we wanted to be able to visually compare the actual test set plot versus the predicted one, which required magnitude from a regression approach. The only requirement to run the notebook (model_notebooks/stock_prediction.ipynb) that contains the code for this model is moving the updated_tickers.txt file from the data folder in the root of the repository to the Google Collab runtime environment.

![Image](https://github.com/user-attachments/assets/ea9d8657-fc80-4017-be95-daa586e3b29f)

## LSTM Model Results
Model Evaluation (Results):
To evaluate the model, we wrote a custom algorithm to test the LSTM model versus the random strategy. We used the same base algorithm for both strategies with slightly different logic to create an accurate measure of performance. The first key simplification for the algorithm was deciding that stocks can only be bought in one hundred dollar increments. The second  simplification was deciding that once a model buys a stock it only sells it at the very last day in the sequence (or when the LSTM model predicts the price will go down to curb losses). If a stock is predicted to go down by the LSTM the selling price for the 100 dollars is reduced proportionally to the stock change. Both the LSTM and random models have the same number of opportunities to buy and sell to maintain precision. After testing both strategies over several trials, we observed that the LSTM on average made eighty dollars more than the random selection of buying and selling options. This method of model evaluation was added in addition to the initial analysis because in a real world setting how much money the model could potentially make is paramount to a company/customer. The code for the model evaluation can be found in model_notebooks/model_evaluation.ipynb. To reproduce our results, the test_stocks directory inside of the data directory must be added to the environment. In addition, the h5 file for the LSTM model inside of the saved_models directory must be added to Collab’s runtime environment.

## Videos
### Combined Video:
[![Watch the video](https://img.youtube.com/vi/pEzSNc4SH6U/0.jpg)](https://www.youtube.com/watch?v=ArFrPVoCB2Y)


### Project goals and challenges [link](https://github.com/ColeJMitchell/machine-learning-stock-prediction/blob/e692b1b6a108eabc829caf21a9d5c02c19da2b24/media/project_problem.mp4):
https://github.com/user-attachments/assets/5966b3ad-78ca-48e2-b37f-1d2733c69b96

### Overview of data collection [link](https://github.com/ColeJMitchell/machine-learning-stock-prediction/blob/main/media/DataCollectionNotebookOverview.mp4):
https://github.com/user-attachments/assets/804e38a0-81fd-4d67-a802-059461bee602

### Overview of sentiment analysis [link](https://github.com/ColeJMitchell/machine-learning-stock-prediction/blob/main/media/SentimentAnalysisNotebookOverview.mp4):
https://github.com/user-attachments/assets/47c3e3bf-1c93-4f0e-8f75-b10799cd7f80

### Stock data collection and preprocessing [link](https://github.com/ColeJMitchell/machine-learning-stock-prediction/blob/main/media/data_collection.mp4):
https://github.com/user-attachments/assets/584fa63e-a2d8-42e2-9a4d-6be22e602dee

### Data split and model architecture [link](https://github.com/ColeJMitchell/machine-learning-stock-prediction/blob/main/media/data_sets_and_model_architecture.mp4):
https://github.com/user-attachments/assets/825df98c-babc-4417-8dca-37b59570d723

### Model training and testing [link](https://github.com/ColeJMitchell/machine-learning-stock-prediction/blob/main/media/training_and_testing.mp4):
https://github.com/user-attachments/assets/edbdf418-3c7e-4596-9c6c-7fee7128137c

### Model evaluation [link](https://www.youtube.com/watch?v=pEzSNc4SH6U):
[![Watch the video](https://img.youtube.com/vi/pEzSNc4SH6U/0.jpg)](https://www.youtube.com/watch?v=pEzSNc4SH6U)
