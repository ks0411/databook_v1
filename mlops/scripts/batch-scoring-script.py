import json
import logging
import os

import joblib
from databook.data_book import DataBook


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
    logging.info("Init complete")


def run(mini_batch):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info(f"Request received")

    results = []
    for raw_data_path in mini_batch:
        dBook = DataBook()
        dBook.load_file(raw_data_path)
        dBook.pre_process_data()
        df = dBook.get_inconsistent_cells()

        result = model.predict(df)
        # results.append({k: v for (k, v) in zip(df.index, result)})
        jsonLine = {}
        jsonLine['FileName'] = str(raw_data_path).split('/')[-1]
        jsonLine['Inference'] = {k: str(v) for (k, v) in zip(df.index, result)}

        results.append(json.dumps(jsonLine))
    logging.info("Request processed")

    return results
