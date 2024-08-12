import pandas as pd

import logging
from utils.sql_data_queries import TrainDatesHandler
from utils.config import PARAMS, MODEL_METRICS, TRAIN_PARAMS
import os
import numpy as np
import tempfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
class FraudDataProcessor:
    """processes transactions data as datafram from the sql engine to a dataframe
    handles the data pre-processing for training, retraining and predictions"""

    # pipeline instances:
    label_enc = LabelEncoder()

    # replace missing values with a constant text, then encode to numeric classes and scale
    state_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="online")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler()),
        ]
    )

    # replace missing values with zero, then encode and scale
    zero_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler())
        ]
    )

    # implement number scaler on numerical features (no missing values)
    # implement text replacement to state and errors
    # implement zero replacement to zip, city and chip
    transformer = ColumnTransformer(
        transformers=[
            ("number_scaler", StandardScaler(), [0, 1, 2, 3, 4, 5, 7, 11, 13, 14]),
            ("NAN_replace_text", state_pipe, [9, 12]),
            ("NAN_replace_zero", zero_pipe, [6, 8, 10]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    def __init__ (self):
        # load data from sql db
        self.sql_handler = TrainDatesHandler()
        self.df = self.sql_handler.get_transactions_to_date()
        
        # save the date of last training, for splitting the data (last month is for prediction):
        self.last_training_date = self.sql_handler.last_training_date
        self.train_df = self.df.loc[self.df['time_stamp'] < self.last_training_date]#
        self.train_df.drop(columns=['id', 'time_stamp'], inplace=True)
        
        self.pred_df = self.df.loc[self.df['time_stamp'] >= self.last_training_date]
        self.pred_df.drop(columns=['id', 'time_stamp', 'is_fraud'], inplace=True) 
        
        # save  a df of prediction dataset that will be presented to user
        self.present_df = self.df.loc[self.df['time_stamp'] >= self.last_training_date]
        self.present_df.drop(columns=['time_stamp', 'is_fraud'], inplace=True) 
        
    def x_y_generator(self):
        """ passes the training dataframe through pipeline to fit to the model:
        split to x, y, normalize data, label y and set bias."""

        # shuffle the data: 
        self.train_df = self.train_df.sample(frac = 1)
        # split to x, y
        xtrain = self.train_df.drop(columns=['is_fraud'])  
        ytrain = self.train_df[['is_fraud']]
                
        # apply transormer
        self.transformer.fit(xtrain)
        xtrain = self.transformer.transform(xtrain)
        ytrain = self.label_enc.fit_transform(ytrain)
        self.xpred = self.transformer.transform(self.pred_df)

        # set output bias
        neg, pos = np.bincount(self.label_enc.transform(self.df['is_fraud']))
        self.output_bias = np.log([pos / neg])

        #reshape labels tensor for tensorflow:
        logging.info(f'reshape ytrain to: {ytrain.shape}')
        ytrain = ytrain.reshape(ytrain.shape[0], 1)
        self.train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        
        
    # @property
    # def output_bias(self):
    # # update output bias:
    #     return self.output_bias
    
def update_params_output_bias(params, data: FraudDataProcessor):
    """update the output bias in the external params, 
    according to the data features for the purpose of training
    with the relevant output bias"""
    params['output_bias'] == data.output_bias
    

def load_saved_model(name="mymodel.keras"):
    version = (len(os.listdir('saved_model')))
    return tf.keras.models.load_model(f'saved_model/{version}/mymodel.keras')

def load_model_weights(model, train_params):
    path = os.path.join(tempfile.mkdtemp(), 'initial_weights.weights.h5')
    train_params['initial_weights'] = path
    initial_weights = path
    logging.info(os.path.exists(path))
    
    if os.path.exists(path):
        logging.info(os.path.exists(path))
        logging.info("initial weights loaded")
        
    else: 
        model.save_weights(train_params['initial_weights'])        
        logging.info('weights saved')
        
    model.load_weights(initial_weights)
    return model

def model_trainer(model, data, params, train_params, output_bias_generator=True, callback=None):
    """ loads the model architecture for new training, with the new
    data for the relvant period"""
    if output_bias_generator:
        output_bias = tf.keras.initializers.Constant(params['output_bias']) 
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_prc',
        verbose=0,
        patience=train_params['patience'],
        mode='max',
        restore_best_weights=True
    )

    callback_list = [early_stopping]

    # add Tensorboard for checking, but there is no validation 
    if callback:
        callback_list.append(callback)
    
    if params['output_bias'] is None:
        model.layers[-1].bias.assign([0.0])
    
    train_ds = data.train_ds.batch(train_params['batch_size']).prefetch(2)

    model.fit(
        train_ds,
        batch_size=train_params['batch_size'],
        epochs=train_params['epochs'],
        callbacks=callback_list
    )
    # save the new model as the latest version
    if not os.path.exists('saved_model'): 
        os.mkdir('saved_model')
    version = str(len(os.listdir('saved_model')) +1)
    os.mkdir(f'saved_model/{version}')
    model.save(f"saved_model/{version}/mymodel.keras")
    return model

def predict(model, data, threshold=0.5):
    """gets model predictions and returns in a df in a human readable format"""
    predictions = model.predict(data.xpred)
    labels = predictions >= threshold
    logging.info(f'labels{labels.shape}')
    results_df = data.present_df
    logging.info(f'results info: {results_df.info}')
    results_df['is_fraud'] = labels

    return results_df
  
def pipeline():
    data = FraudDataProcessor()
    data.x_y_generator()

    update_params_output_bias(PARAMS, data)
    model = load_saved_model()
    print(model.summary())

    model = load_model_weights(model, PARAMS)
    new_trained_model = model_trainer(model, data, PARAMS, TRAIN_PARAMS)
    return predict(new_trained_model, data)

if __name__ == "__main__":
    results = pipeline()
    print(results.head(10))
    print(results.tail(10))


