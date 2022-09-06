# DataBook demo

The repository is made of the following files and folders:

- *data_book.py* provides a class for data preparation, that is, converts an input JSON representation of an Excel file to features.
- *Model-Training.ipynb* provides the code for training a model. Note this requires providing a file to support labeling.
- *Data-Inference.ipynb* provides the code for doing inference over a cell or sheet of paper. At the moment we are using the same data used for training to show this.
- *model.pkl* is a trained model that can be used for inference.
- *unit_test_databook_class.ipynb* provides unit tests for the *DataBook* class
- *Data-v1* holds some sample data

## Limitations

We are not providing the code to deploy these artifacts to the cloud.
We are only supporting vertical consistency.
Training has been made on a very limited set of data.
Training has been made over very few ranges of cells with a single inconsistent cell.
We are not considering formatting information to produce fratures.

## Running the code

The code should be run on a local machine or any cloud notebook platform. 
Have an environment with Pandas and Scikit-Learn.
