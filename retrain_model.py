from utils.sql_data_queries import TrainDatesHandler
from utils.config import REGISTERED_MODEL_NAME
from load_from_drive import gcp_auth_download
import utils.data_prep_pipeline as data_prep_pipeline
import os, shutil
import json

#gcp_auth_download()
database = os.getenv('DATABASE')

# get the current version number of the model (excluding the .yaml file)
current_version = len(os.listdir(f'mlruns/models/{REGISTERED_MODEL_NAME}')) -1      
print(f'current version: {current_version}')
# restart re-training period if the period for prediction (1/2019-3/2020) is over:  
if current_version == 15:
    versions_list = range(2,16)
    for v in versions_list:
        path = f'mlruns/models/{REGISTERED_MODEL_NAME}/version-{v}'
        try:
            shutil.rmtree(path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    current_version =1
    
#get dates list: 
sql_handler = TrainDatesHandler()
if sql_handler.dates_df.shape[0] < current_version:
    print('you are up to date')
    pass
elif current_version == 0: 
    eval_results = data_prep_pipeline.re_train_pipeline()    
    print(json.dumps(eval_results))
else:
    date = sql_handler.dates_df['train_date'].iloc[current_version]
    print(date)    
    eval_results = data_prep_pipeline.re_train_pipeline(date=date)    
    print(json.dumps(eval_results))

