#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,recall_score
from sklearn.model_selection import cross_val_score,train_test_split,StratifiedKFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# In[2]:


train = pd.read_csv('/kaggle/input/novartis/Dataset/train.csv')
test = pd.read_csv('/kaggle/input/novartis/Dataset/test.csv')
sub = pd.read_csv('/kaggle/input/novartis/Dataset/sample_submission.csv')


# In[3]:


train.head(5)


# In[4]:


len(train)


# In[5]:


train.dtypes


# In[6]:


train['DATE'] = pd.to_datetime(train['DATE']) #Converting date to type of datetime


# In[7]:


train.isnull().sum() #checking for missing values


# In[8]:


pd.value_counts(train['X_12'])


# In[9]:


train['X_12'].isnull().sum()


# In[10]:


import seaborn as sns


# In[11]:


sns.countplot('X_12',hue='MULTIPLE_OFFENSE',data=train)


# In[12]:


x_12_1 = train[train['MULTIPLE_OFFENSE']==1].X_12


# In[13]:


x_12_1.isnull().sum()


# In[14]:


np.mean(x_12_1)


# In[15]:


train['X_12'].fillna(1,inplace=True) #replacing missing values with 1 as most of the values are 1


# In[16]:


train.isnull().sum()


# In[17]:


train['X_12'] = train['X_12'].astype('int64') #converting to int64


# In[18]:


train.dtypes


# In[19]:


train_t = train.drop('INCIDENT_ID',axis=1)


# In[20]:


train_t.head(5)


# In[21]:


X = train_t.iloc[:,0:-1]
y = train_t.iloc[:,-1]


# In[22]:


#Creating separate columns for year,month, and day


# In[23]:


X['YEAR'] = X.DATE.dt.year 
X['MONTH'] = X.DATE.dt.month
X['DAY'] = X.DATE.dt.day


# In[24]:


X = X.drop('DATE',axis=1)


# In[25]:


X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 0
                                                    )


# In[26]:


random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))


# In[27]:


kfold = StratifiedKFold(n_splits=10)


# In[28]:


cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y_train , scoring = "recall", cv = kfold, n_jobs=4))


# In[29]:


cv_results


# In[30]:


cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())


# In[31]:


cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting"]})


# In[32]:


cv_res


# In[33]:


test.head(5)


# In[34]:


test['DATE'] = pd.to_datetime(test['DATE'])


# In[35]:


test['YEAR'] = test.DATE.dt.year
test['MONTH'] = test.DATE.dt.month
test['DAY'] = test.DATE.dt.day


# In[36]:


test['X_12'].fillna(1,inplace=True)


# In[37]:


test_t = test.drop('DATE',axis=1)


# In[38]:


test['X_12'] = test['X_12'].astype('int64')


# In[39]:


test_t['X_12'] = test['X_12']


# In[40]:


test_t = test_t.drop('INCIDENT_ID',axis=1)


# In[41]:


test_t.head(5)


# In[42]:


gb = GradientBoostingClassifier()
parameters = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05],
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "n_estimators":[10]
    }
gbSCV = GridSearchCV(gb,parameters, cv=kfold, scoring="recall", n_jobs= -1, verbose = 1)
gbSCV.fit(X , y)


# In[44]:


gbSCV_best = gbSCV.best_estimator_

# Best score
gbSCV.best_score_


# In[45]:


pred3 = gbSCV_best.predict(test_t)


# In[46]:


sub1 = sub


# In[49]:


sub1 = pd.DataFrame({"INCIDENT_ID":test['INCIDENT_ID'],"MULTIPLE_OFFENSE":pred3})


# In[50]:


len(sub1)


# In[51]:


sub1.to_csv('sub4.csv',index=False)


# In[52]:


pred2 = gbSCV_best.predict(test_t)


# In[53]:


sub1 = pd.read_csv('/kaggle/input/novartis/Dataset/sub1.csv')


# In[54]:


sub2 = sub1


# In[55]:


sub2['MULTIPLE_OFFENSE'] = pred2


# In[56]:


sub2.to_csv('sub2.csv',index=False)


# In[ ]:




