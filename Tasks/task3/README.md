# Task 3. Classification model with custom Transformers and Estimators. Identification of survivors based on data on passengers of the Titanic.

## Description

This task is similar to Task 1. 
You will need to solve the same problem with the passengers of the Titanic, but now you have to build a model using custom Transformers and Estimators.
The folder `data/titanic/` contains data about the passengers of the Titanic. 
It will be necessary to build a model that will determine whether a person survived the crash or not.
It will be necessary to build a model using custom Transformers and Estimators.

## Data

The folder `data/titanic/` contains data about the passengers of the Titanic.
The data is divided into 2 sets: training (`train.csv`) and test (`test.csv`).
The `Survived` column was cut from the test data set. 
This column is located in the file `is_survived.csv`.

The data contains the following fields:

    |-------------|--------------------------------------------|------------------------------------------------|
    | Variable    | Definition                                 | Key                                            |
    |-------------|--------------------------------------------|------------------------------------------------|
    | PassengerId | Unique identifier                          |                                                |
    | survival    | Survival                                   | 0 = No, 1 = Yes                                |
    | pclass      | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
    | sex         | Sex                                        |                                                |
    | Age         | Age in years                               |                                                |
    | sibsp       | # of siblings / spouses aboard the Titanic |                                                |
    | parch       | # of parents / children aboard the Titanic |                                                |
    | ticket      | Ticket number                              |                                                |
    | fare        | Passenger fare                             |                                                |
    | cabin       | Cabin number                               |                                                |
    | embarked    | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |
    |-------------|--------------------------------------------|------------------------------------------------|

A more detailed description of the data can be found on [Kaggle](https://www.kaggle.com/c/titanic).

## Task

The task is to train a machine learning model that will predict whether the passenger survived the crash or not based on the presented data. 

Conditions:
* Model should be trained using custom extensions of the Spark ML transformer and estimator classes and integration of Python libraries. You can use the skeleton code prepared for you in SparkCustomMLPipeline folder.
* You need to train the model on a training data set `train.csv`. 
* You need to test the model on a test dataset `test.csv`.
* Calculate metrics based on the file `is_survived.csv`. 
* Compare the results of two models.