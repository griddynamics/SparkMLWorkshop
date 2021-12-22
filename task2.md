# Task 2. Clustering model. The definition of a private university or not.

## Description

The folder `data/university/` contains university data.
It will be necessary to cluster these universities into private and non-private.
It will be necessary to build a clustering model using the k-means algorithm or another clusterization algorithm from the Spark ML library.

## Data

The folder `data/university/` contains university data (`train.csv`).
The `Private` column was cut from the data set.
This column is located in the file `is_private.csv`.

The data contains the following fields:

    |-------------|-------------------------------------------------------------------------|
    | Variable    | Definition                                                              |
    |-------------|-------------------------------------------------------------------------|
    | Name        | University name                                                         |
    | Private     | A factor with levels No and Yes indicating private or public university |
    | Apps        | Number of applications received                                         |
    | Accept      | Number of applications accepted                                         |
    | Enroll      | Number of new students enrolled                                         |
    | Top10perc   | Pct. new students from top 10% of H.S. class                            |
    | Top25perc   | Pct. new students from top 25% of H.S. class                            |
    | F.Undergrad | Number of fulltime undergraduates                                       |
    | P.Undergrad | Number of parttime undergraduates                                       |
    | Outstate    | Out-of-state tuition                                                    |
    | Room.Board  | Room and board costs                                                    |
    | Books       | Estimated book costs                                                    |
    | Personal    | Estimated personal spending                                             |
    | PhD         | Pct. of faculty with Ph.D.’s                                            |
    | Terminal    | Pct. of faculty with terminal degree                                    |
    | S.F.Ratio   | Student/faculty ratio                                                   |
    | perc.alumni | Pct. alumni who donate                                                  |
    | Expend      | Instructional expenditure per student                                   |
    | Grad.Rate   | Graduation rate                                                         |
    |-------------|-------------------------------------------------------------------------|

A more detailed description of the data can be found on [Kaggle](https://www.kaggle.com/karthickaravindan/k-means-clustering-project/).

## Task

The task is to train a machine learning clustering model that will split university into 2 clusters (private and not private).

Conditions:
* Model should be trained only using Spark ML clusterization algorithm.
* You need to train the model on a training data set `train.csv`.
* Calculate metrics based on the file `is_private.csv`.
