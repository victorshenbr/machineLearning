import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier as gbm
from scipy.sparse import csr_matrix, hstack


pd.options.display.max_rows = 50
pd.options.display.max_columns = 999

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
def plotCol(df, col):
    val = df[col].value_counts()
    l = range(len(val))
    plt.bar(l, val)
    plt.xticks(l,val.keys(), rotation='vertical')

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

def main():
    dir = 'C:/gitproject/machineLearning/'
    #df = pd.read_table(dir + "train.tsv")
    #tests = pd.read_table(dir + "test.tsv")
    df = pd.read_table('../input/train.tsv', engine='c')
    tests = pd.read_table('../input/train.tsv', engine='c')

    df['price'] = np.log1p(df['price'])
    trainRowCount = df.shape[0]
    trainY = df['price']
    del df['price']
    merge: pd.DataFrame = pd.concat([df, tests])
    submission: pd.DataFrame = tests['test_id']
    del df
    del tests

    fillMissingVal(merge)
    reshapeData(merge)
    colToCat(merge)

    minTermFrequency = 10
    # transform name to tfidf
    cv = CountVectorizer(min_df = minTermFrequency)
    X_nameTermFrequency = cv.fit_transform(merge['name'])
    cvCat = CountVectorizer()
    X_catTermFrequency = cvCat.fit_transform(merge['category_name'])
    maxDescriptCount = 50000
    tv = TfidfVectorizer(min_df=maxDescriptCount, ngram_range=(1,3), stop_words='english')
    X_descriptionFrequency = tv.fit_transform(merge['item_description'])

    lb = LabelBinarizer(sparse_output=True)
    x_brandName = lb.fit_transform(merge['brand_name'])

    # compress sparsed categorical matrix to CSR
    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],sparse=True).values)

    # create train parameters from
    sparse_merge = hstack((X_dummies, X_nameTermFrequency, X_catTermFrequency, X_descriptionFrequency, x_brandName)).tocsr()

    x_trainDf = sparse_merge[:trainRowCount]
    x_test = sparse_merge[trainRowCount:]


    model = Ridge(solver="sag", fit_intercept=True, random_state=500)
    model.fit(x_trainDf, trainY)
    preds = model.predict(X=x_test)
    submission['price'] = np.expm1(preds)
    submission.to_csv("submission_ridge.csv", index=False)

if __name__ =="__main__":
    main()
