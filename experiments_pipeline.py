from data_prep_pipeline import FraudDataProcessor, update_params_output_bias, load_model_weights
from utils.config import PARAMS
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(asctime)s - %(levelname)s - %(message)s"
)

class TrainPipeline(FraudDataProcessor):
    
    def __init__(self, date=None):
        super().__init__(date)

    # what i have from parent: 
    # self.sql_handler
    # self.retrain_df.drop(columns=['id', 'time_stamp'], inplace=True)
    # self.predict_df.drop(columns=['time_stamp', 'is_fraud'], inplace=True)
    # self.ypred = self.predict_df['is_fraud'].drop(columns=['id', 'time_stamp', 'is_fraud'], inplace=True)
    
    def training_data_generator(self):
        """
        split to featrures and labels, then splits again to 
        eventually receive (train, val, test).
        creates tensorflow.Dataset train_ds, val_ds, test_ds, as class attributes
        """
        X = self.retrain_df.drop(columns=['is_fraud'])
        Y = self.retrain_df['is_fraud']
        X_train, xtest, Y_train, ytest = train_test_split(
            X,
            Y,
            test_size=0.2,  # test size default 25%
            random_state=42,
            shuffle=True,
            stratify=Y, # maintain class balance between splits
        )
        xtrain, xval, ytrain, yval = train_test_split(
            X_train,
            Y_train,
            test_size=0.25,
            random_state=42,
            shuffle=True,
            stratify=Y_train,
        )
        # apply label encoder on labels:
        ytrain = self.label_enc.fit_transform(ytrain)
        yval = self.label_enc.fit_transform(yval)
        ytest = self.label_enc.transform(ytest)

        # apply pipeline on feature values
        self.transformer.fit(xtrain)
        xtrain = self.transformer.transform(xtrain)
        xval = self.transformer.transform(xval)
        xtest = self.transformer.transform(xtest)
        
        self.xpred = xtest
        # set initial bias
        neg, pos = np.bincount(self.label_enc.transform(Y))
        self.initial_bias = np.log([pos / neg])

        # reshape labels tensor to fit tf requirements: 
        ytrain = ytrain.reshape(ytrain.shape[0], 1)
        yval = yval.reshape(yval.shape[0], 1)
        ytest = ytest.reshape(ytest.shape[0], 1)
        logging.info('test data split successfully')

        self.train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        self.val_ds = tf.data.Dataset.from_tensor_slices((xval, yval))
        self.test_ds = tf.data.Dataset.from_tensor_slices((xtest, ytest))

        logging.info('ready for training')


if __name__ == '__main__':
    data = TrainPipeline()
    data.training_data_generator()

    print(data.retrain_df.head())
    print(data.val_ds)
    print(len(data.val_ds))
