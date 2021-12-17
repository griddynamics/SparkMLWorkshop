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
    line = line.replace('[', '')
    line = line.replace(']', '')
    line = line.replace('\n', '')
    line = re.split('[,]', line)

    line_dict = {}
    for i, value in enumerate(line):
        if i != len(line) - 1:
            name = "feature_" + str(i)
        else:
            name = "label"
        line_dict[name] = value
    rows.append(line_dict)

# Initialize a dataframe from the list
df = pd.DataFrame(rows)

feature_columns = []
for i in range(0, len(df.columns) - 1):
    feature_columns.append("feature_" + str(i))
label_column = "label"

model = LogisticRegression()
model.fit(df[feature_columns], df[label_column])
model_string = base64.b64encode(pickle.dumps(model)).decode('utf-8')

# Output to stdin, so that rdd.pipe() can return the string to pipedRdd.
print(model_string)
