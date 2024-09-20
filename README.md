# Fraud Analysis Prediction and Automation
XXXXXXXXX add links to different parts
descriptions 
instructions
details
## Description
This is a show case project for developing and using the machine learning systems at the various stages of machine learning life cycle.

Trained MLP neural network that predicts fraud credit card transactions, based on the IBM Credit card Transactions Database.

The UI that allows the user to log in and watch monthly predictions for the fraud cases.  

## Structure and tools
The project includes the following steps and tools: 
- Exploratory Data Analysis (mainly in notebooks with Pandas)
- Data conversion to sql with SQLAlchemy, and data pre-processing
- A basic MLP binary-classifier with Keras
- Model experiments tracking, and model registry with MlFlow
- User interface to call predictions, using Flask, Plotly-Dash-Table, SQL-Alchemy and a minimal bootstrap 
- Data storage on Google Drive
- Re-training automation with git-actions workflow, triggered on a schedule to update the model with new monthly data. Currently set to every hour, for convenience. 

## Use Case, storyline and details: 
This project uses the IBM Credit Card Fraud Transactions Dataset. The dataset contains 6 years of transactions with the lables Fraud or not_fraud. Data for 4 years has been used for the project, starting 1.1.2017, due to the size of the file.
The model is initially trained until 31.12.2018, and retrains through the workflow on additional monthly data.

The UI includes viewing rights for two predefined users: Pinkey and Brain.
The user credentials for the predictions view


## Instructions:
clone the repo

install the requirements:
`pip install -r requirements.txt`

**note:** make sure that tensorflow version is  <2.15, otherwise the logging with mlflow can be problematic. 

To automatically re-clone the repo and get the updated data after every retrain, run: 
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

To run the UI: 
`python app.py` 

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





install google cloud https://cloud.google.com/sdk/docs/install
tensorflow < 2.15 


project structure
how to reset database dates

