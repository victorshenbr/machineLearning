import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

dir = 'C:/gitproject/machineLearning/'
df = pd.read_table(dir + "train.tsv")

df.describe()
df.head(5)

# check summary statistics of null columns
df.isnull().sum()
totRow = df.count()['train_id']

df['splited_categories'] = df.apply(lambda row: str(row["category_name"]).split("/"), axis = 1)
maxLength = max([len(x) for x in df["splited_categories"].tolist()])
curCat = 1

# make new column for each category
while curCat <= maxLength:
    df["cat_" + str(curCat)] = df.apply(lambda row: row['splited_categories'][curCat - 1] \
        if (len(row['splited_categories']) >= curCat) else None, axis = 1)
    curCat += 1

uniqueCat = list(set([item for sublist in df['splited_categories'].tolist() for item in sublist]))
len(uniqueCat)

