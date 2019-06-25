# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:45:32 2018

@author: srika
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import scale


def predict(l):
    # finds the userIds corresponding to the top 5 similarities
    # calculate the prediction according to the formula
    return (df[l.index] * l).sum(axis=1) / l.sum()


# use userID as columns for convinience when interpretering the forumla
df = pd.read_csv('ratings.csv').pivot(columns='userId',
                                                index='movieId',
                                                values='rating')

similarity = pd.DataFrame(cosine_similarity(
    scale(df.T.fillna(0))),
    index=df.columns,
    columns=df.columns)
#print(similarity)
# iterate each column (userID),
# for each userID find the highest five similarities
# and use to calculate the prediction for that user,
# use fillna so that original ratings dont change

res = df.apply(lambda col: ' '.join('{}'.format(mid) for mid in (0 * col).fillna(
    predict(similarity[col.name].nlargest(6).iloc[1:])).nlargest(5).index))
#print(res)
#print(res.index)
#res[['userId', 'movieId1', 'movieId2', 'movieId3', 'movieId4', 'movieId5']] = res[:, 0].str.split(',\s+', expand=True)
df = pd.DataFrame.from_items(zip(res.index, res.str.split(' ')))
df = df.transpose()
#print(df)
df.columns = ['movie-id1', 'movie-id2', 'movie-id3', 'movie-id4', 'movie-id5']
df['customer_id'] = df.index
df = df[['customer_id', 'movie-id1', 'movie-id2', 'movie-id3', 'movie-id4', 'movie-id5']]
df.to_csv('output.txt', sep=' ', index=False, header=False)
#df.to_csv('output.txt', sep=' ', index=False)

#print(res)
#res = res.apply(pd.to_numeric, errors='ignore')
#print(res.iloc[:,1])
#df.to_csv('output.txt', sep=' ', index=False)
