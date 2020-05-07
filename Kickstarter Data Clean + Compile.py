#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import pandas as pd
import seaborn as sns

df = pd.read_csv('Sample_Kickstarter_Projects_2019.csv')


# In[3]:


print(df.columns)
df.drop(['source_url', 'urls', 'photo', 'currency_symbol', 'currency_trailing_code', 
         'country_displayable_name', 'usd_type', 'Unnamed: 0', 
          'converted_pledged_amount', 'pledged', 'location'], axis = 1, inplace = True)
df.drop(['state'], axis = 1, inplace = True)


# In[4]:


import re

#create new columns

df['usd_goal'] = df['goal'] * df['static_usd_rate']
#df['goal_reached'] = df['usd_pledged'].apply(lambda x: 1 if x >= df['usd_goal'] else 0)

df['spotlight'] = df['spotlight'].apply(lambda x: 1 if x == True else 0)
df['staff_pick'] = df['staff_pick'].apply(lambda x: 1 if x == True else 0)
df['is_starrable'] = df['is_starrable'].apply(lambda x: 1 if x == True else 0)
df['disable_communication'] = df['disable_communication'].apply(lambda x: 1 if x == True else 0)
#we can use state for a multiclassification problem later


#columns about length of time
df['duration_seconds'] = df['deadline'] - df['launched_at']
df['duration_days'] = df['duration_seconds'] / 3600

#columns about blurb length, number capitals
df['blurb_length'] = df['blurb'].str.len()
df['blurb']=df['blurb'].fillna('None')
def caps(series):
    #return len(re.findall(r'[A-Z]',series))
    count = len([letter for letter in series if letter.isupper()])
    return count
df['blurb_caps'] = df['blurb'].apply(caps)

#column of title number capitals
df['name_length'] = df['name'].str.len()
df['name_caps'] = df['name'].apply(caps)

#extract info from creator
creator_df = df['creator'].str.split(',', expand = True)
df['creator'] = creator_df[0].apply(lambda x: x.replace('{"id":', ''))

val = df['creator'].value_counts()
df['repeat_creator'] = df['creator'].apply(lambda x: 1 if val.loc[x]>1 else 0)


# In[5]:


df


# In[6]:


#extract 2 new columns from 'category column'
split_df = df['category'].str.split(',', expand = True)
display(split_df)
def extract_category(series):
    series = series.split(':')
    series1 = series[1].split('/')[0].replace('"', '')
    return series1

df['overarching_category'] = split_df[2].apply(extract_category)
split_df[2] = split_df[2].apply(extract_category)
def extract_position(series):
    series = series.split(':')
    return int(series[1])
df['page_position'] = split_df[3].apply(extract_position)
df.drop(['category'], axis = 1, inplace = True)
df.loc[df['usd_pledged'] >= df['usd_goal'], 'goal_reached'] = 1
df.loc[df['usd_pledged'] < df['usd_goal'], 'goal_reached'] = 0


# In[7]:


import matplotlib.pyplot as plt
corr = df.corr()
plt.subplots(figsize=(11,11))
sns.heatmap(corr, vmax=0.9,cmap="YlGnBu", square=True)


# In[8]:


df.drop(['is_starred', 'is_backing', 'permissions', 'friends'], axis = 1, inplace = True)


# In[9]:


null_vals = df.isnull().sum()
null_vals.sort_values(ascending = False)

df['blurb_length'] = df['blurb_length'].fillna(0)


# In[10]:


df.to_csv('Kickstarter Data Cleaned and Engineered.csv')


# In[11]:


df


# In[ ]:




