from utils.sql_data_queries import TrainDatesHandler
import data_prep_pipeline
import os
import json

database = os.getenv('DATABASE')

# def update_last_training_date_in_db():
current_version = len(os.listdir('mlruns/models/fraud_analysis')) -1

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
    eval_results = data_prep_pipeline.re_train_pipeline(date=date)#, model_version=current_version+1)    
    print(json.dumps(eval_results))

