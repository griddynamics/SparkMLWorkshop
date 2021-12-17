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
# model = ???

# Here we keep input data to Dataframe constructor
rows = []

# Iterate over standard input
for line in sys.stdin:
    # Parse here your input and append result into rows
    # ???

    rows.append(line_dict)

# Initialize a dataframe from the list
df = pd.DataFrame(rows)

# Run inference
pred = model.predict(df)

df['preds'] = pred

print(df)
