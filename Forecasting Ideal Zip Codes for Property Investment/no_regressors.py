#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import numexpr
#load data
#from google.colab import files
#uploaded=files.upload()


# In[4]:
numexpr.set_num_threads(24)

import pandas as pd
data = pd.read_csv('Zip_Zhvi_SingleFamilyResidence.csv',encoding='ISO-8859-1' )


# In[5]:


#drop the columns from 1996 because we're only looking at 1997 - present 
complete = data.drop(['1996-04', '1996-05', '1996-06', '1996-07', '1996-08', '1996-09',
                             '1996-10', '1996-11','1996-12'],axis=1)


# In[6]:


#drop the rows with Nans 
#get a list of dates for the subset 
na_list = list(complete.columns)
na_list = na_list[7:]
complete = complete.dropna(subset=na_list)
complete


# In[7]:


#preprocess the original complete dataset
#drop the other vars for melting
complete_clean=complete.drop(['Metro', 'RegionID','City', 'State','SizeRank', 'CountyName'], axis=1)

#melt dataset
complete_melt = pd.melt(complete_clean, id_vars=['RegionName'])
#format the date 
complete_melt['variable']=pd.to_datetime(complete_melt['variable'], format='%Y-%m')


# In[8]:


complete_melt.head()
#add back the metro and other vars for matching
complete_join = pd.merge(complete_melt, complete[['Metro', 'City', 'State', 'SizeRank', 'RegionID','RegionName']], how='inner', on='RegionName')



# In[ ]:

#!conda install -c conda-forge fbprophet
#import prophet 
from fbprophet import Prophet
#rename dates and values to ds and y
complete_join=complete_join.rename(columns={"variable":"ds","value":"y"} )


# In[9]:


#break up into a sets of data for each zipcode
df_sets = {}
for i in set(complete_join['RegionName']):
  print(i)
  df = complete_join[complete_join['RegionName'] == i]
  df = df.reset_index(level=0)
  df_sets[i]= df

#verify 
print("the length is" , (len(df_sets)))







# In[14]:


#create a model for each zipcode 
prophet_models = {}
for i,df in df_sets.items():
  prophet_models[i] = Prophet(interval_width=0.95)
  prophet_models[i].fit(df)
  print (len(prophet_models))
print ("the length is", len(prophet_models))


# In[ ]:


#get future date for each set, 13 months in advance -- arbitrary choice 
future_dates = {}
for i,df in prophet_models.items():
  future_dates[i] = prophet_models[i].make_future_dataframe(periods=13, freq='M')



# In[ ]:


#create the forecast 
forecasts = {}
for i,df in df_sets.items():
  forecasts[i] = prophet_models[i].predict(future_dates[i])
  print(len(forecasts))
  print(forecasts[i][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# In[ ]:


#plot the models 
import matplotlib.pyplot as plt
for i,model in prophet_models.items():
  model.plot(forecasts[i], uncertainty=True)
  plt.ylabel('Median Housing Average (in dollars)')
  plt.xlabel('Date')
  plt.title(i)
plt.show()


# In[ ]:


#how do I know which ones are the best models?
#find the Mean Absolute Percentage Error -- pick the smallest one 

MAPE_all= {}
for key in df_sets:
  y_value = df_sets[key]['y'].astype('float')
  y_hat = forecasts[key]['yhat'].astype('float')
  MAPE_all[key]=(np.mean(np.abs((y_value-y_hat)/y_value))*100)


# In[ ]:


#find the smoothest graph
#https://www.w3resource.com/python-exercises/dictionary/python-data-type-dictionary-exercise-1.php
#for dictionary sort 
import operator 
sorted_dict = sorted(MAPE_all.items(), key=operator.itemgetter(1))

#get the top 5 
print ("RegionName", "MAPE")
sorted_dict[:10]


# In[ ]:


first_10 = sorted_dict[:10]


# In[ ]:

import matplotlib.pyplot as plt
first_10_keys = []
for i in range(0, len(first_10)):
   first_10_keys.append(first_10[i][0])


for i in first_10_keys:
    prophet_models[i].plot(forecasts[i], uncertainty=True)
    plt.ylabel('Median Housing Average (in dollars)')
    plt.xlabel('Date')
    plt.title(i)
plt.show
