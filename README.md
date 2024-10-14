# Fraud Analysis Prediction and Automation
This is a show case project for developing and using the machine learning systems at the various stages of ML lifecycle.

- [description](#description)
- [instructions](#instructions)
- [structure and tools](#structure-and-tools)
- [Use Case, storyline and details](#use-case-storyline-and-details)

## Description
The core model is an MLP neural network that predicts fraudulant credit card transactions, based on the [IBM Credit card Transactions Database](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions?resource=download&select=credit_card_transactions-ibm_v2.csv).
The model is automatically retrained on a monthly basis, based on new data, using a scheduled Github Actions workflow.
The project includes a UI that allows authorized users to log in and vie monthly predictions for potential fraud.  

## Structure and Tools
The project consists of the following steps, using different ML tools and frameworks: 
- **Exploratory Data Analysis (EDA)**: primarily using Jupyter notebooks and Pandas
- **Data management**: Data is converted and stored as SQL, using SQLAlchemy.
- **Data pre-processing**: written in Python classes.
- **Modeling**: A binary-classifier MLP neural network with Keras
- **Experiment tracking and model registry** Model performance and registry is tracked by MlFlow
- **User interface (UI)**: Built using Flask and Plotly-Dash-Table, with minimal Bootstrap styling. 
- **Data storage** the data files are stored on Google Drive
- **Automation** The model is automatically retrained on new data, using Github Actions workflow, triggered on a schedule. Currently set hourly, for demo purposes. 

## Use Case, storyline and details: 
This project uses the IBM Credit Card Fraud Transactions Dataset, which contains 6 years of transactions with the labels Fraud or not_fraud. Data for 4 years has been used for the project, starting 1.1.2017, due to the size of the file.
The model is initially trained until 31.12.2018, and retrains through the workflow on additional monthly data.

The UI includes viewing rights for two predefined users: user1 and user2. The user credentials for the predictions view only in the Project Presentation that was submitted. 


## Usage Instructions:
1. Install Docker
2. Install git-lfs 
3. create a project folder in the desired location. for example: mkdir ******
4. clone the repo: *******link********* (pay attention: the repo is relatively large)
5. Pull the init.sql file using git-lfs: `git lfs pull`
6. Pull the images from Docker hub:
    a. `docker pull triggertiger/model_production:postgres` (might be that 'sudo' is required in case that permission is denied)
    b. `docker pull triggertiger/model_production:latest`
7. Run: `docker compose -f docker-compose.yml up`
8. In the first run, building the database might take several minutes. 
9. The app is running on localhost, on port 8080. go to http://127.0.0.1:8080 in your browser, follow the 

## Reproduction: 

2. create a virtual environment and download the requirements
    `python3.9 -m venv model2production`
    `source model2production/bin/activate`
    `pip install -r requirements_mlflow.txt`

    **note:** make sure that tensorflow version is  <2.15, to avoid potential issues with MLFlow logging. 

WhatDoWeDoTonight

3. Download the [dataset](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions/data?select=credit_card_transactions-ibm_v2.csv) from Kaggle and save it under `/data/6y_ibm.csv`
    > You can use the kaggle-cli tool for this with kaggle-cli tool (installed with the requirements), by running: 
    'kg dataset -u <username> -p <password> -o <owner> -d <dataset>`. 
4. Make sure you have Postgres running
4. Run: data/set_database.sh which will execute:
    - data/clean_csv.py
    - data/db_population.py
    - data/db_setup.py
5. In order to run experiments, make sure to update the environment variables in the .env file. you can run: `python experiments_pipeline.py`
5. You're all set! 
    for experiments with new model parameters, run experiments_pipeline.py. 
    Continue as with the user instructions. 
