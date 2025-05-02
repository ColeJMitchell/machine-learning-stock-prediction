# Machine Learning Stock Prediction
- Author(s): Cole Mitchel, Ava Ginsberg, and Benjamin Gregory
- Date: 2025-05-02

## Overview
_

## Usage and Setup
The enclosed material includes `./requirements.txt`, containing the relevant packages and versions for the enclosed code. +

It is recommend that you open a virtual python environment and install packages within, avoiding possible conflicts with local data. The program installs several materials locally, including but not limited to the NLTK (Natural Language ToolKit) corpus data. For these local installs, we open directories at the project root; however, these directories are not pushed to the public branch. With this in mind, run any provided code with caution as to not overwrite local data. Additionally, the source code leverages the public reddit data api. The user must include a `.env` file at the root of this project with the following contents,
```
CLIENT_SECRET= <provided by reddit>
CLIENT_ID= <provided by reddit>
USERNAME= <provided by user for reddit>
PASSWORD= <proided by user for reddit>
KAGGLE_USERNAME = <Kaggle API username>
KAGGLE_KEY = <Kaggle API key>
```
These environmental attributes are retrieved prior to collecting reddit submission and comment data. Note that reddit makes no guarantee regarding consistent rate limits for the public API. In some cases, reddit API access may not be available due to circumstances on their end. Aditionally, the collected reddit data is stored in the `./data/` directory, but may not be available in the public branch for this project.

## Data Sources and Structure
Whether the raw data sources are included in the public repository or are generated throughout the usage, notice that there are numerous data files included in the `./data/` directory. These files are generated as checkpoints, throughout the modeling process. Description of the data files, origins, and usage are listed below. 

| Filename (`./data/<filename>`) | Source | Desc |
|---|---|---|
| `posts-with-tickers.parquet` |  |  |
| `wallstreetbets-comment-collection-wss.parquet` |  |  |
| `wallstreetbets-comment-collection.parquet` |  |  |
| `wallstreetbets-collection-wss.parquet` |  |  |
| `wallstreetbets-collection.parquet` |  |  |
| `kaggle-reddit-wsb.parquet` |  |  |
| `reddit_wsb.txt` |  |  |
| `tickers.txt` |  |  |
| `updated_tickers.txt` |  |  |

## Workflow

