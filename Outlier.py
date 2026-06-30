import pandas as pd
import numpy as np
from scipy import stats
# scores = {
#     'Scores':[55, 60, 58, 72, 65, 68, 63, 70, 57, 61, 98, 66]
# }
# df = pd.DataFrame(scores)
# # print(df)
# q1 = df['Scores'].quantile(0.25)
# q2 = df['Scores'].quantile(0.50)
# q3 = df['Scores'].quantile(0.75)
# # print(q1,q2,q3)
# # print(q1,q2,q3)
# iqr = q3-q1
# print(iqr)
# lower = q1-1.5*iqr
# higher = q3 + 1.5*iqr
# print(lower)
# print(higher)
# df_cleaned = df[(df['Scores']>=lower)&(df['Scores']<=higher)]
# print(df_cleaned)
#
# # Z-Score
# mean = df['Scores'].mean()
# std = df['Scores'].std()
# print(mean,std)
# outlier = [x for x in df['Scores'] if abs((x-mean)/std)>2]
# print(outlier)
#
import matplotlib.pyplot as plt
data = {
  'order_id': list(range(1, 21)),
  'amount'  : [450,520,480,510,490,530,470,500,2500,515,488,502,478,525,495,505,485,512,50,498],
  'quantity': [2,3,2,3,2,4,2,3,25,3,2,3,2,4,2,3,2,3,1,0]
}
df = pd.DataFrame(data)
# print(df)
q1 = df['amount'].quantile(0.25)
q2 = df['amount'].quantile(0.50)
q3 = df['amount'].quantile(0.75)
print(q1,q2,q3)
IQR = q3-q1
print(IQR)
lower = q1 - 1.5*IQR
higher = q3 + 1.5*IQR
print(lower,higher)
outlier = df[(df['amount']<lower)|(df['amount']>higher)]
print(outlier)
df_cleaned = df[(df['amount'] >= lower) & (df['amount'] <= higher)]
print(df_cleaned)

# x = [10,55,56,75,80,14,12,16,19,20,52]
# print(lambda i:)