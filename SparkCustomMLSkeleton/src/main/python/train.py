#!/usr/bin/python3

import pandas as pd
import pickle
import sys
import base64
import re

from sklearn.linear_model import LogisticRegression

# Here we keep input data to Dataframe constructor
rows = []

for line in sys.stdin:
    # Parse here your input and append result into rows
    # ???

# Initialize a dataframe from the list
df = pd.DataFrame(rows)

feature_columns = []
for i in range(0, len(df.columns) - 1):
    feature_columns.append("feature_" + str(i))
label_column = "label"

model = LogisticRegression()
# Fit your model here
# ???

# Convert model into string (you can use base64 encode)
# model_string = ???

# Output to stdin, so that rdd.pipe() can return the string to pipedRdd.
print(model_string)
