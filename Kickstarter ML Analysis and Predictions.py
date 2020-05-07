#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[3]:


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df = pd.read_csv('Kickstarter Data Cleaned and Engineered.csv')
df.drop(['blurb', 'creator', 'currency', 'deadline', 'id', 'name', 'Unnamed: 0', 'created_at', 'state_changed_at',
         'profile', 'slug', 'launched_at', 'duration_seconds','goal',
         'static_usd_rate','spotlight', 'country'], axis = 1, inplace = True)

#test different models of ML and see which ones work well
    #tested RFC, GBC
#extract key features and communicate what they mean
# tv = df['goal_reached'].value_counts()
# print(tv)
# print(tv[0]/(tv[0]+tv[1]))
# sns.distplot(df['goal_reached'])


# In[4]:


# Code to understand skew

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
newdf = df.select_dtypes(include=numerics)

fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(20, 120))
plt.subplots_adjust(right=2)
plt.subplots_adjust(top=2)
sns.color_palette("husl", 8)
for i, feature in enumerate(list(newdf.columns), 1):
    if(feature=='MiscVal'):
        break
    plt.subplot(len(list(newdf.columns)), 3, i)
    sns.distplot(newdf[feature])
    plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)
    
    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
    
    plt.legend(loc='best', prop={'size': 10})
        
plt.show()


# In[5]:


fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
plt.subplot(3, 3, 1)
sns.distplot(df['name_caps'])
plt.subplot(3, 3, 2)
sns.distplot(df['page_position'])
plt.subplot(3, 3, 3)
sns.distplot(df['blurb_caps'])


# In[6]:


#code to fix skew in key variables

df['name_caps'] = np.log1p(df['name_caps'])
df['page_position'] = np.log1p(df['page_position'])
df['blurb_caps'] = np.log1p(df['blurb_caps'])
# max_blurb = np.max(df['blurb_length'])
# df['blurb_length'] = np.log1p((1+max_blurb)-df['blurb_length'])
# sns.distplot(df['blurb_length'])
print('done')


# In[22]:



#columns that arent useful for ML: 'blurb', 'creator', 'currency', 'deadline', 'id', 'name', 'profile', 'slug'

train_df = pd.get_dummies(df)
X = train_df.drop(['goal_reached','usd_pledged', 'backers_count' ], axis = 1)
print(X.columns)
#, 'usd_pledged', 'backers_count' 
y = train_df['goal_reached']
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
#need more feature engineering / feature analysis
print('done')


# In[8]:


#PCA


# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score
#another metric we might want is precision = low false positive rate. 
#You don't want to invest money according to the model only for it not to work.
#create models

# parameters = {'n_estimators': [10], 'max_depth': [None,15,20,25]}
# clf = GridSearchCV(RandomForestClassifier(), param_grid = parameters)
# print('grid compiled')
# clf.fit(X_train_scaled, y_train)
# print('fit complete')
# print(clf.best_params_)

clf = RandomForestClassifier(n_estimators = 10, max_depth = 20).fit(X_train_scaled,y_train)
print('RFC Score: ', clf.score(X_test_scaled, y_test))
y_predict = clf.predict(X_test_scaled)
print('RFC Precision Score: ', precision_score(y_test, y_predict))

plot_confusion_matrix(clf, X_test_scaled, y_test,
                                 cmap=plt.cm.Blues)
plt.show()

# for i in [.01, .1, .5]:
#     clf2 = GradientBoostingClassifier(learning_rate = i).fit(X_train_scaled, y_train)
#     print('GBC Score: ' ,clf2.score(X_test_scaled, y_test), 'learning_rate: ', i)

print(' ')


# In[10]:


# parameters2 = {'learning_rate': [.1, .5, .8]}
# clf2 = GridSearchCV(GradientBoostingClassifier(), param_grid = parameters2, scoring = 'precision')
# print('grid compiled')
# clf2.fit(X_train_scaled, y_train)
# print('fit complete')
# print(clf2.best_params_)


clf2 = GradientBoostingClassifier(learning_rate = .8).fit(X_train_scaled, y_train)
y_predict2 = clf2.predict(X_test_scaled)
print('GBC Score: ', clf2.score(X_test_scaled, y_test))
print('GBC Precision Score: ', precision_score(y_test, y_predict2))


# In[54]:


importances1 = pd.Series(clf2.feature_importances_, X.columns)
importances1 = importances1.sort_values(ascending = False)
plt.bar(importances1.head(10).index, importances1.head(10))
plt.xticks(rotation = 90)
plt.title('Gradient Boosted Classifier: Feature Importances')


# In[13]:


# from sklearn.neighbors import KNeighborsClassifier


# for i in np.arange(1,10):
#     clf3 = KNeighborsClassifier(n_neighbors = i)
#     clf3.fit(X_train_scaled, y_train)
#     y_predict3 = clf.predict(X_test_scaled)
#     print('KNN Score: ' ,clf3.score(X_test_scaled, y_test), 'n_neighbors: ', i)
#     print('KNN Precision Score: ', precision_score(y_test, y_predict3))


# In[53]:


# #next steps:
# research similar projects and see what we can add

#create 2 models? The question is:
    #what factors lead to making a kickstarter better
    #text analysis?
    #category importance


# In[27]:


# importances
# for (x, y) in zip(importances.index, importances):
#     print(x)
#     print(y)

X_train_scaled = pd.DataFrame(data = X_train_scaled, columns = X.columns)


# In[52]:


import shap
shap_values = shap.TreeExplainer(clf2).shap_values(X_train_scaled)
# shap.summary_plot(shap_values, X_train_scaled, plot_type = 'bar')


# In[57]:



def ABS_SHAP(df_shap,df):
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True).tail(25)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=(5,6),legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    plt.title('Key Features')
    
ABS_SHAP(shap_values,X_train_scaled)


# In[51]:


# Get the predictions and put them with the test data.
X_test_scaled = pd.DataFrame(data = X_test_scaled, columns = X.columns)
X_output = X_test_scaled.copy()
X_output.loc[:,'predict'] = np.round(clf.predict(X_output),2)

# Randomly pick some observations
random_picks = np.arange(1,1000,50) # Every 50 rows
S = X_output.iloc[random_picks]
S
# Initialize your Jupyter notebook with initjs(), otherwise you will get an error message.
shap.initjs()

# Write in a function
def shap_plot(j):
    explainerModel = shap.TreeExplainer(clf2)
    shap_values_Model = explainerModel.shap_values(S, check_additivity=False)
    p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], S.iloc[[j]])
    return(p)

shap_plot(0)


# In[60]:


df['usd_goal'].median()


# In[ ]:




