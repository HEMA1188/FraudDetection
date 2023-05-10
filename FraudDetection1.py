#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter(action='ignore')
import pickle


# In[2]:


df=pd.read_csv('Fraud.csv')
df.head()


# In[3]:


df.duplicated()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df.groupby('isFlaggedFraud').size()


# In[8]:


df.groupby('isFraud').size()


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings(action='ignore')


# In[10]:


# Preprocessing Libraries
from sklearn.preprocessing import RobustScaler


# In[11]:


# Model training libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from collections import Counter
from imblearn.under_sampling import NearMiss               # Undersampling
from imblearn.over_sampling import RandomOverSampler       # Oversampling
from imblearn.combine import SMOTETomek                    # Both Undersampling & Oversampling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# For checking acuracy
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[12]:


#visualization
df[df['isFraud']==1]


# In[13]:


# Countplot of 'type'
plt.figure(figsize=(7,3))
plt.title('type vs count')
sns.countplot(data=df,x='type',palette='coolwarm')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()


# In[14]:


plt.figure(figsize= (20,10))
sns.heatmap(df.corr(),  annot = True,cmap= "cubehelix_r")
plt.show()


# In[15]:


# Plotting subplot for amount and time column
fig, ax = plt.subplots(1, 2, figsize=(18,4))
amount_val = df['amount'].values
time_val = df['step'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Step', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])
plt.show()


# In[16]:


# Countplot of 'isFraud'
plt.figure(figsize=(7,3))
plt.title('isFraud vs count')
sns.countplot(data=df,x='isFraud')
plt.xlabel('isFraud')
plt.ylabel('Count')
plt.show()


# Note: Imbalancing Data

# In[17]:


df['isFraud'].value_counts()


# In[18]:


# Let's look at the percentage of each category in isFraud column(target column)
print("No Frauds:",df['isFraud'].value_counts()[0]/len(df['isFraud'])*100)
print("Frauds:",df['isFraud'].value_counts()[1]/len(df['isFraud'])*100)


# In[19]:


numerical=['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']


# In[20]:


# Boxplot for each variable in numerical list
def boxplots_visual(data,column):
    fig, ax = plt.subplots(2,3,figsize=(18,6))
    fig.suptitle('Boxplot for each variable',y=1, size=20)
    ax=ax.flatten()
    for i,feature in enumerate(column):
        sns.boxplot(data=data[feature],ax=ax[i], orient='h')
        ax[i].set_title(feature+ ', skewness is: '+str(round(data[feature].skew(axis = 0, skipna = True),2)),fontsize=15)
        ax[i].set_xlim([min(data[feature]), max(data[feature])])
boxplots_visual(data=df,column=numerical)
plt.tight_layout()


# In[21]:


# Checking nameOrig,nameDest column
nameOrig=df['nameOrig'].unique()
print("Unique in nameOrig:",len(nameOrig))
print(nameOrig)

nameDest=df['nameDest'].unique()
print("Unique in nameDest:",len(nameDest))
print(nameDest)


# In[22]:


# Checking isFlaggedFraud column
df['isFlaggedFraud'].value_counts()


# In[23]:


# Dropping columns that are not needed
df.drop(['nameOrig','nameDest','isFlaggedFraud'],axis=1,inplace=True)


# In[24]:


# Applying onehot encoding on type column
df=pd.get_dummies(data=df,columns=['type'],drop_first=True)
df.head()


# In[25]:


# We are using RobustScaler to scale down the numerical features as RobustScaler is less prone to outliers
scale=RobustScaler()
for feature in numerical:
    df[feature]=scale.fit_transform(df[feature].values.reshape(-1, 1))
df.head()


# In[26]:


# Splitting our data into independent and dependent features
x=df.drop('isFraud',axis=1)
y=df['isFraud']


# In[27]:


x.columns


# In[28]:


df[df['isFraud']==1]


# In[29]:


# Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(x,y)
print(model.feature_importances_)


# In[32]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# In[33]:


# Doing train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,train_size=0.7)


# In[34]:


# Applying StratifiedKFold
skf=StratifiedKFold(n_splits=3, shuffle=False, random_state=None)
lr=LogisticRegression()
param={'C':10.0 **np.arange(-1,1)}
test1=RandomizedSearchCV(lr,param,cv=skf,n_jobs=-1,scoring='accuracy')
test1.fit(X_train,y_train)


# In[35]:


y_pred=test1.predict(X_test)
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
print('\n')
print("Accuracy Score: \n",accuracy_score(y_test,y_pred))
print('\n')
print("Classification Report: \n",classification_report(y_test,y_pred))


# Training with RandomForestClassifier by setting class_weight

# In[36]:


test2=RandomForestClassifier(class_weight={0:1,1:100})
test2.fit(X_train,y_train)


# In[37]:


y_pred=test2.predict(X_test)
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
print('\n')
print("Accuracy Score: \n",accuracy_score(y_test,y_pred))
print('\n')
print("Classification Report: \n",classification_report(y_test,y_pred))


# NOte: RandomForestClassifier has better results

# In[38]:


#UnderSampling

ns=NearMiss()
X_train_ns,y_train_ns=ns.fit_resample(X_train,y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))


# In[39]:


model1=RandomForestClassifier()
model1.fit(X_train_ns,y_train_ns)


# In[40]:


y_pred=model1.predict(X_test)
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
print('\n')
print("Accuracy Score: \n",accuracy_score(y_test,y_pred))
print('\n')
print("Classification Report: \n",classification_report(y_test,y_pred))


# In[41]:


#OverSampling

from imblearn.over_sampling import RandomOverSampler
os=RandomOverSampler()
X_train_ns,y_train_ns=os.fit_resample(X_train,y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))


# In[ ]:


model2=RandomForestClassifier()
model2.fit(X_train_ns,y_train_ns)


# In[ ]:


y_pred=model2.predict(X_test)
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
print('\n')
print("Accuracy Score: \n",accuracy_score(y_test,y_pred))
print('\n')
print("Classification Report: \n",classification_report(y_test,y_pred))


# Confusion Matrix: 
#  [[1906192     144]
#  [    444    2006]]
# 
# 
# Accuracy Score: 
#  0.9996919508001421
# 
# 
# Classification Report: 
#                precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00   1906336
#            1       0.93      0.82      0.87      2450
# 
#     accuracy                           1.00   1908786
#    macro avg       0.97      0.91      0.94   1908786
# weighted avg       1.00      1.00      1.00   1908786

# In[43]:


# open a file, where you want to store the data
file = open('fraud_prediction.pkl', 'wb')

# dump information to that file
pickle.dump(model2, file)
file.close()


# In[ ]:


myfile = open('fraud_prediction.pkl','rb')
mymodel = pickle.load(myfile)
myprediction = mymodel.predict(X_test)
print("Confusion Matrix: \n",confusion_matrix(y_test,myprediction))
print('\n')
print("Accuracy Score: \n",accuracy_score(y_test,myprediction))
print('\n')
print("Classification Report: \n",classification_report(y_test,myprediction))


# In[ ]:




