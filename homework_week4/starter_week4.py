#!/usr/bin/env python
# coding: utf-8

# In[21]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[22]:


import pickle
import pandas as pd
import statistics


# In[23]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[24]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[25]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-02.parquet')


# In[26]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[27]:


sd = round(statistics.stdev(y_pred), 2)


# ## Q1. Notebook
# What's the standard deviation of the predicted duration for the February 2022 dataset?

# In[28]:


print(f"Answer: The standard deviation is {sd}.")


# ## Q2. Preparing the output
# What's the size of the output file?

# In[34]:


year = 2022
month = 2

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[35]:


df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

output_file = f"output/ride_prediction_{year:04d}-{month:02d}.parquet"

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[37]:


get_ipython().system('ls -lh output/')


# Answer: The size of the output file is 57M.

# ## Q3

# In[ ]:


get_ipython().system('jupyter nbconvert --to script starter_week4.ipynb')

