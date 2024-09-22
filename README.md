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

The UI includes viewing rights for two predefined users: Pinkey and Brain. The user credentials for the predictions view only in the Project Presentation that was submitted. 


## Instructions:
1. clone the repo

2. install the requirements:
    `pip install -r requirements.txt`

    **note:** make sure that tensorflow version is  <2.15, to avoid potential issues with MLFlow logging. 

3. To automatically re-clone the repo and get the updated data after every retrain, run: 
    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

4. Run the UI locally: 
    run: `python app.py` 
    On your internet browser, go to https:localhost:8080, 
    log in as one of the authorized users, and follow the instructions. 

## To do 
create a cron job to pull the repo
Make a workflow cycle - delete the model versions
fix the bootstrap UI
fix flash messages on UI
make MLFLOW user login
Project flowchart
Project tree
add here an image
Make git on schedule
Add env.example

do I have to install google cloud https://cloud.google.com/sdk/docs/install???
 




