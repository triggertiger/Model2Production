import mlflow.keras.save
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import os
import tempfile
import datetime
import logging
from utils.config import DATA_PATH, ORIGINAL_CSV, PARAMS, MODEL_METRICS, TRAIN_PARAMS, MLFLOW_URI

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.keras

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

mlflow.set_tracking_uri(uri=MLFLOW_URI)

class FraudDataProcessor:
    # set pipeline instances:
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

    def __init__(self, csv_path, end_date = None):
        self.data = pd.read_csv(csv_path)
        self.end_date = end_date
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.initial_bias = 0.0

        
    def data_df_prep(self):
        """processes the dataframe columnss to desired format.
        returns sorted dataframe. if end date is set, returns dataframe
        until end date"""

        logging.info("pre-processing training data") 

        # rename columns
        self.data.rename(str.lower, axis="columns", inplace=True)
        self.data.rename(
            columns={
                "use chip": "use_chip",
                "merchant name": "merchant_name",
                "merchant city": "merchant_city",
                "merchant state": "merchant_state",
                "errors?": "errors",
                "is fraud?": "is_fraud",
            },
            inplace=True,
        )

        # convert amount to float
        self.data["amount"] = self.data["amount"].str[1:].astype("float64")

        # set time series index for sorting
        self.data[["hour", "minute"]] = self.data["time"].str.split(":", expand=True)
        self.data["date"] = pd.to_datetime(
            self.data[["year", "month", "day", "hour", "minute"]]
        )
        self.data.set_index("date", inplace=True)
        self.data.sort_index(inplace=True)

        self.data.drop(columns=["time"], inplace=True)

        if self.end_date:
            logging.info("period end date found. Sorting data to relevant dates")
            self.data = self.data.loc[: self.end_date]
            return self.data
        return self.data

     

    def data_splitter(self):#, update_bias=True):
        """takes fraud dataframe, performs train_test_split and applies scaling and encoding to features and labels
        performs initial bias calculation for the imbalanced dataset.
        returns xtrain, ytrain, xval, yval, xtest, ytest and initial bias"""

        # split to x (features) and y (labels), and split twice (train, val, test)
        X = self.data.drop(columns=["is_fraud"])
        y = self.data[["is_fraud"]]
        Xtrain, xtest, Ytrain, ytest = train_test_split(
            X,
            y,
            test_size=0.2,  # test size default 25%
            random_state=42,
            shuffle=True,
            stratify=y,
        )
        xtrain, xval, ytrain, yval = train_test_split(
            Xtrain,
            Ytrain,
            test_size=0.25,
            random_state=42,
            shuffle=True,
            stratify=Ytrain,
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
        neg, pos = np.bincount(self.label_enc.transform(y))
        self.initial_bias = np.log([pos / neg])

        # reshaping labels tensor to fit the model requirements of 2 dimensions
        ytrain = ytrain.reshape(ytrain.shape[0], 1)
        yval = yval.reshape(yval.shape[0], 1)
        ytest = ytest.reshape(ytest.shape[0], 1)
        logging.info('test data split successfully')

        self.train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        self.val_ds = tf.data.Dataset.from_tensor_slices((xval, yval))
        self.test_ds = tf.data.Dataset.from_tensor_slices((xtest, ytest))

        logging.info('ready for training')

def update_output_bias(params, bias: float):
    params['output_bias'] = bias
    return params['output_bias']

def model_generator(params, metricas, output_bias_generator=True):
    output_bias = params['output_bias']  
    logging.info(f'output bias calculated: {output_bias}') 

    # setting an outpubias for the model, based on the data
    if output_bias_generator:
        output_bias = tf.keras.initializers.Constant(output_bias) 
    
    # build model input layer and dense layers:
    model = keras.Sequential([keras.Input(shape=params['train_feature_size'])])
    for lay in range(params['nr_of_layers']):
        layer = keras.layers.Dense(
            params['layer_size'][lay],
            activation=params['activation1'],
            name=f'Denselayer{lay+1}'
        )
        model.add(layer)

   # add dropout, activation for the output layer        
    model.add(keras.layers.Dropout(params['dropout']))
    model.add(keras.layers.Dense(1, activation='sigmoid',
                           bias_initializer=output_bias
                           ))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss=keras.losses.BinaryCrossentropy(),               
        metrics=list(metricas.values())
    )
    logging.info("model generated")
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
        print(callback_list)
    # load or create initial weights
    try:
        initial_weights = train_params['initial_weights']
        logging.info("initial weights loaded")
    except KeyError:
        train_params['initial_weights'] = os.path.join(tempfile.mkdtemp(), 'initial_weights.weights.h5')
        model.save_weights(train_params['initial_weights'])
        logging.info('weights saved')
        initial_weights = train_params['initial_weights']
        logging.info(f'initial weights created: {initial_weights}')

    model.load_weights(initial_weights)

    logging.info("begin training")
    # eliminate use of output-bias (to balance classes)
    if params['output_bias'] is None:
        model.layers[-1].bias.assign([0.0])
    
    # prepare data with Tensorflow Datasetbatches: 
    train_ds = data.train_ds.batch(train_params['batch_size']).prefetch(2)
    val_ds = data.val_ds.batch(train_params['batch_size']).prefetch(2)
    test_ds = data.test_ds.batch(train_params['batch_size']).prefetch(1)

    model.fit(
        train_ds,
        batch_size=train_params['batch_size'],
        epochs=train_params['epochs'],
        validation_data=val_ds,
        verbose=1,
        callbacks=callback_list
    )

    return model

def mlflow_run(name, params, train_params, metricas, data, run_name=None):
    
    tags = {k: v for k, v in params.items()}
    mlflow.set_experiment(name)
    experiment = mlflow.get_experiment_by_name(name)
    client = mlflow.tracking.MlflowClient()     # what is my tracking ?
    run = client.create_run(experiment.experiment_id)
    mlflow.tensorflow.autolog(disable=True)

    with mlflow.start_run(run_name=run_name) as run:
        logging_callback = mlflow.tensorflow.MlflowCallback(run)
        mlflow.log_params(params)
        mlflow.log_params(train_params)
        mlflow.set_tags(tags)

        model = model_generator(params=params, metricas=metricas)
        print(model.summary())
        model = model_trainer(model, data, params, train_params, logging_callback)
        test_ds = data.test_ds.batch(train_params['batch_size']).prefetch(1)   
        print(model.summary())         # explanation about prefetch:https://stackoverflow.com/questions/63796936/what-is-the-proper-use-of-tensorflow-dataset-prefetch-and-cache-options
        logging.info(f'test_ds shape: {len(data.test_ds)}, {data.test_ds}')
        logging.info(f'train ds shape: {len(data.train_ds)}, {data.train_ds}')

        results = model.evaluate(test_ds, verbose=1)#, batch_size=train_params['batch_size'], verbose=1)
        predictions=model.predict(test_ds) #, batch_size=train_params['batch_size'])

        for name, value in zip(model.metrics_names, results):
            print(name, ': ', value)
        
        logging.info(f'prediction scores:\n\n {predictions}')
        mlflow.tensorflow.log_model(model, "models")
        if not os.path.exists('saved_model'): 
            os.mkdir('saved_model')
        version = str(len(os.listdir('saved_model')) +1)
        os.mkdir(f'saved_model/{version}')
        model.save(f"saved_model/{version}/mymodel.keras")
        #mlflow.keras.save_model(model, f"saved_model/{version}", save_exported_model=False, )

        cm = ConfusionMatrixDisplay.from_predictions(
            np.concatenate([y for x, y in test_ds], axis=0), predictions > 0.2
        )
        mlflow.log_figure(cm.figure_, 'test_confusion_matrix.png')
        
def load_saved_model(name="mymodel.keras"):
    version = (len(os.listdir('saved_model')))
    return tf.keras.models.load_model(f'saved_model/{version}/mymodel.keras')

def allocate_predictions(predictions, threshold=0.5):
    fraud_indices = predictions >= threshold   # bool
    return fraud_indices

def reverse_transformer(transformer, x):
    logging.info('reverse transformer starting')
    return transformer.inverse_transform(x)
    


if __name__ == "__main__":
    params = PARAMS
    data = FraudDataProcessor(os.path.join(DATA_PATH, ORIGINAL_CSV))
    #data.end_date = ('2018-01-01')
    data.data_df_prep() 
    data.data_splitter()
    
    params['output_bias'] = data.initial_bias
    #mlflow_run('fraud_detection', params=params, train_params=TRAIN_PARAMS, metricas=MODEL_METRICS, data=data, run_name='larger_layer_take2')
    new_model = load_saved_model()
    new_model.summary()
    
    xpred = data.xpred
    print(xpred.shape)
    predictions = new_model.predict(xpred)
    print(predictions)
    labels = allocate_predictions(predictions)
    logging.info(f'columns: {data.data.columns}')
    print(labels)
    results_df = pd.DataFrame(xpred[:10], columns= data.data.drop(columns=['is_fraud']).columns)
    results_df['is_fraud'] = labels[:10]
    print(results_df.head())
    #pred_transactions = reverse_transformer(data.transformer, data.xpred)
    #print(pred_transactions)

