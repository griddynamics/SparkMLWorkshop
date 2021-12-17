#!/usr/bin/python3

import pandas as pd
import pickle
import sys
import base64
import re

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# Read the model, deserialize and unpickle it.
model = pickle.loads(
          base64.b64decode(
            open("python.model").read().encode('utf-8')
          )
        )

# Here we keep input data to Dataframe constructor
rows = []

# Iterate over standard input
for line in sys.stdin:
    line = line.replace('[', '')
    line = line.replace(']', '')
    line = line.replace('\n', '')
    line = re.split('[,]', line)

    line_dict = {}
    for i, value in enumerate(line):
        name = "feature_" + str(i)
        line_dict[name] = value

    rows.append(line_dict)

# Initialize a dataframe from the list
df = pd.DataFrame(rows)

# Run inference
pred = model.predict(df)

df['preds'] = pred

print(df)
