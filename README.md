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


## Usage Instructions:
1. Install Docker
2. Install git-lfs 
3. create a project folder in the desired location. for example: mkdir ******
4. clone the repo: 
4. 

2. create a virtual environment and download the requirements
    `python3.9 -m venv model2production`
    `source model2production/bin/activate`
    `pip install -r requirements.txt`

    **note:** make sure that tensorflow version is  <2.15, to avoid potential issues with MLFlow logging. 

3. You need to update an .env file, based on `env.example`, with the relevant example variables, available in the project presentation file.

4. Download the data file: run: `curl -L "https://drive.usercontent.google.com/download?id=${DRIVE_FILE_ID}&confirm=xxx" -o ./tmp/fraud_transactions.db`
    ***the curly brackets {} need to stay in the command***

5. To automatically pull the repo and get the updated data after every retrain, run: 
    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

6. Run the UI locally: 
    run: `python app.py` 
    On your internet browser, go to http://127.0.0.1:8080, 
    log in as one of the authorized users, and follow the instructions. 

## To do 

add set_database bash code file. including adding gitignore for db files. 
            update env.example
            make MLFLOW user login
            Make a workflow cycle - delete the model versions
            fix the bootstrap UI
            fix flash messages on UI
            create a python file for 'make 6 years to 4 years' 

Project flowchart
Project tree

do I have to install google cloud https://cloud.google.com/sdk/docs/install???
 

## Reproduction:
1. Clone the repository
2. Install requirements `pip install -r requirements.txt`
3. Download the [dataset](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions/data?select=credit_card_transactions-ibm_v2.csv) from Kaggle and save it under `/data/6y_ibm.csv`
    > You can use the kaggle-cli tool for this with kaggle-cli tool (installed with the requirements), by running: 
    'kg dataset -u <username> -p <password> -o <owner> -d <dataset>`. 
4. Run: utils/set_database.sh which will execute:
    - utils/clean_csv.py
    - utils/db_population.py
    - utils/db_setup.py
5. In order to run experiments, make sure to update the environment variables in the .env file. 
5. You're all set! 
    for experiments with new model parameters, run experiments_pipeline.py. 
    Continue as with the user instructions. 
