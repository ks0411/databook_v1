import os
import logging
import json
import numpy
import joblib
from data_book import *

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model.pkl"
    )
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    # logging.info("Init complete")

def run(raw_data_path):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    # logging.info("Request received")
    # data = json.loads(raw_data)["data"][0]['path']
    # raw_data_path = Input(type='uri_file', path=data)
    
    dBook = DataBook()
    dBook.load_file(raw_data_path)
    dBook.pre_process_data()
    df=dBook.get_inconsistent_cells()
    # print(df.index)

    results = model.predict(df)
    return {k:v for (k,v) in zip(df.index, results)}
    # logging.info("Request processed")
    # return result.tolist()