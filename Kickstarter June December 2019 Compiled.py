#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df7 = pd.read_csv('Kickstarter July 2019 Data.csv')
df7['Month'] = 7
df8 = pd.read_csv('Kickstarter August 2019 Data.csv')
df8['Month'] = 8
df9 = pd.read_csv('Kickstarter September 2019 Data.csv')
df9['Month'] = 9
df10 = pd.read_csv('Kickstarter October 2019 Data.csv')
df10['Month'] = 10
df11 = pd.read_csv('Kickstarter November 2019 Data.csv')
df11['Month'] = 11
df12 = pd.read_csv('Kickstarter December 2019 Data.csv')
df12['Month'] = 12

df = pd.concat([df7, df8, df9, df10, df11, df12], ignore_index = True)

display(df.shape)

df = df.sample(frac = .45)
print("sample done")
df = df.sort_values(by = ['Month'])
df = df.drop_duplicates('id', keep = 'last')
print('deleted duplicates')
df.to_csv('Sample_Kickstarter_Projects_2019.csv')
print('download finished')


# In[ ]:





# In[ ]:




