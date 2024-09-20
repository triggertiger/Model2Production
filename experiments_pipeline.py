from data_prep_pipeline import FraudDataProcessor, update_params_output_bias, load_model_weights
from utils.config import MLFLOW_URI, PARAMS, TRAIN_PARAMS, MODEL_METRICS, EXPERIMENT_NAME
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import tempfile
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow import keras


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(asctime)s - %(levelname)s - %(message)s"
)

class TrainPipeline(FraudDataProcessor):
    
    def __init__(self, date=None):
        super().__init__(date)
        
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
        self.output_bias = np.log([pos / neg])

        # reshape labels tensor to fit tf requirements: 
        ytrain = ytrain.reshape(ytrain.shape[0], 1)
        yval = yval.reshape(yval.shape[0], 1)
        ytest = ytest.reshape(ytest.shape[0], 1)
        logging.info('test data split successfully')

        self.train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        self.val_ds = tf.data.Dataset.from_tensor_slices((xval, yval))
        self.test_ds = tf.data.Dataset.from_tensor_slices((xtest, ytest))

        logging.info('ready for training')

def model_generator(data, params, model_metrics, output_bias_generator=True):
    params['output_bias'] = data.output_bias
    logging.info(f'output bias calculated: {data.output_bias}')

    # set an output bias for the model, based on the data class imbalance
    if output_bias_generator:
        output_bias = tf.keras.initializers.Constant(data.output_bias)

    # build model: nr of layers by the params setting
    model = keras.Sequential([keras.Input(shape=params['train_feature_size'])])
    for lay in range(params['nr_of_layers']):
        layer = keras.layers.Dense(
            params['layer_size'][lay],
            activation=params['activation1'],
            name=f'Denselayer{lay+1}'
        )
        model.add(layer)
    
    # add dropout, activation:
    model.add(keras.layers.Dropout(params['dropout']))
    model.add(keras.layers.Dense(1, activation='sigmoid',
                                 bias_initializer=output_bias))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=list(model_metrics.values())
    )
    logging.info(model.summary())
    return model

def model_trainer(model, data, params, train_params, callback=None):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_prc',
        verbose=1,
        patience=train_params['patience'],
        mode='max',
        restore_best_weights=True
    )
    callback_list = [early_stopping]

    # placeholder for mlflow callback: 
    if callback:
        callback_list.append(callback)

    # load or create initial weights: 
    try: 
        initial_weights = train_params['initial_weights']
        logging.info('initial weights loaded')
    except KeyError:
        train_params['initial_weights'] = os.path.join(tempfile.mkdtemp(), 'initial_weights.weights.h5')
        model.save_weights(train_params['initial_weights'])
        initial_weights = train_params['initial_weights']
        logging.info(f'weights created: {initial_weights}')
    
    model.load_weights(initial_weights)

    logging.info('data created. creating model')

    if params['output_bias'] is None:
        model.layers[-1].bias.assign([0.0])
    
    # set batches: 
    train_ds = data.train_ds.batch(train_params['batch_size']).prefetch(2)
    val_ds = data.val_ds.batch(train_params['batch_size']).prefetch(2)
    test_ds = data.test_ds.batch(train_params['batch_size']).prefetch(2)
    
    model.fit(
        train_ds,
        batch_size=train_params['batch_size'],
        validation_data=val_ds,
        verbose=1,
        callbacks=callback_list
    )

    return model

def mlflow_experiment_pipeline(exp_name, data, params, train_params, model_metrics, run_name=None):
    uri = MLFLOW_URI
    mlflow.set_tracking_uri(uri)
    
    tags = {k: v for k, v in params.items()}
    
    mlflow.tensorflow.autolog()
    mlflow.set_experiment(exp_name)
    
    with mlflow.start_run(run_name=run_name):
        # mlflow.log_params(params)
        # mlflow.log_params(train_params)
        mlflow.set_tags(tags)
        
        model = model_generator(data, params, model_metrics)
        print(model.summary())
        model = model_trainer(model, data, params, train_params)
        test_ds = data.test_ds.batch(train_params['batch_size']).prefetch(2)
        
        results = model.evaluate(test_ds, verbose=1)
        predictions = model.predict(test_ds) 

        for name, value in zip(model.metrics_names, results):
            print(name, ': ', value)
        logging.info(f'prediction scores:\n\n {predictions}')
        mlflow.tensorflow.log_model(model, "models")
        #generate confusion matrix and chart .png (need to handle the fig closing in thread)
        def get_conf_matrix(test_ds=test_ds):
            """
            generates confusion matrix chart. 
            handles the opening of a figure in a thread and tensorflow.autolog() not recording data
            """
            matplotlib.use('Agg')
            cm = ConfusionMatrixDisplay.from_predictions(
                np.concatenate([y for x, y in test_ds], axis=0), 
                predictions > 0.2
                )
            mlflow.log_figure(cm.figure_, 'test_confusion_matrix.png')
            plt.close(cm.figure_)
            return
        get_conf_matrix()
        return results 
        
def allocate_predictions(predictions, threshold=0.5):
    fraud_indices = predictions >= threshold   # bool
    return fraud_indices

def reverse_transformer(transformer, x):
    logging.info('reverse transformer starting')
    return transformer.inverse_transform(x)

if __name__ == '__main__':
    data = TrainPipeline()
    data.training_data_generator()
    PARAMS['output_bias'] = update_params_output_bias(PARAMS, data)
    #model = model_generator(data, PARAMS, MODEL_METRICS)  
    #model_trainer(model, data, PARAMS, TRAIN_PARAMS)
    results = mlflow_experiment_pipeline(EXPERIMENT_NAME, data, PARAMS, TRAIN_PARAMS, MODEL_METRICS, run_name='one_layer_test')

    
