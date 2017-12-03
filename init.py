import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

pd.options.display.max_rows = 50
pd.options.display.max_columns = 999

dir = 'C:/gitproject/machineLearning/'
df = pd.read_table(dir + "train.tsv")
tests = pd.read_table(dir + "test.tsv")

def splitCats(df):
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

# exploration for categories by count, col is the column name
def plotCol(col):
    val = df[col].value_counts()
    l = range(len(val))
    plt.bar(l, val)
    plt.xticks(l,val.keys(), rotation='vertical')

def vectorization(df):
    colNameList = df.columns
    textCol = df['item_description'].apply(lambda x: str(x))

    #tfidf = TfidfVectorizer( stop_words='english')
    #tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
    #tfs = tfidf.fit_transform(textCol)
    countvec = CountVectorizer()
    tfidfVal = countvec.fit_transform(textCol)

#descDf = pd.DataFrame(tfidfVal.toarray(), columns=countvec.get_feature_names())
def fillMissingVal(df):
    for colName in df:
        df[colName].fillna("missing", inplace = True)

def reshapeData(df):
    brandCount = 5000
    catCount = 1000
    # get column name of all the non missing brand name and keep only the top brand count
    brandRank = df['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:brandCount]
    df.loc[~df["brand_name"].isin(brandRank),'brand_name'] = 'missing'

    # keep only the top cateogry name
    catRank = df['category_name'].value_counts().index[:catCount]
    df.loc[~df['category_name'].isin(catRank), "category_name"] = "missing"

def colToCat(df):
    df['name'] = df['name'].astype('category')
    df['brand_name'] = df['brand_name'].astype('category')
    df['item_condition_id'] = df['item_condition_id'].astype('category')
    df['shipping'] = df['shipping'].astype('category')


def tfidfTransform(df):
    minTermFrequency = 10
    # transform name to tfidf
    cv = CountVectorizer(min_df = minTermFrequency)
    nameTermFrequency = cv.fit_transform(df['name'])

    cvCat = CountVectorizer()
    catTermFrequency = cvCat.fit_transform(df['category_name'])

    maxDescriptCount = 50000
    tv = TfidfVectorizer(min_df=maxDescriptCount, ngram_range=(1,3), stop_words='english')
    descriptionFrequency = tv.fit_transform(df['item_description'])



df['price'] = np.log1p(df['price'])
trainRowCount = df.shape[0]
trainY = df['price']
del df['price']
merge: pd.DataFrame = pd.concat([df, tests])
submission: pd.DataFrame = tests[['test_id']]
del df
del tests



