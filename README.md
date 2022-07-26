# DataBook demo

The repository is made of the following files and folders:

- *data_book.py* provides a class for data preparation, that is, converts an input JSON representation of an Excel file to features.
- *Model-Training.ipynb* provides the code for training a model. Note this requires providing a file to support labeling. We are not providing details about this at the moment.
- *model.pkl* is a trained model
- *scoreing_script_RT* is a scoring script for an Azure ML real-time end-point
- *scoreing_script_BATCH* is a scoring script for an Azure ML batch end-point
- *test_scoring_script_RT.ipynb* provides an example of using a real-time end-point. At the moment this is not supported in the demo environment.
- *unit_test_databook_class.ipynb* provides unit tests for the *DataBook* class
- *Data-v1* holds some sample data

## Limitations

We are not providing a fully working real-time end-point.
We are not providing the code for deployment.
We are only supporting vertical consistency.
Training has been made on a very limited set of data.
Training has been made over ranges of cells with a single inconsistent cell-
We are not considering formatting information to produce fratures.

## Running the code

Create an Azure ML batch-endpoint by using *scoring_script_BATCH* and one *environment* supporting *scikit-learn* and *pandas*.
