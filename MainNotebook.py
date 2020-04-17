#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


import curvefit
from curvefit.pipelines.basic_model import BasicModel
from curvefit.core import functions


# In[4]:


dataset = pd.read_csv('dataset_for_CurveFit.csv')
dataset = dataset.reset_index(drop=True)
dataset = dataset.sort_values(by=["DateI"])
dataset["SE"] = np.random.normal(scale=0.1, size=dataset.shape[0])
dataset["cov_one"] = dataset.shape[0]*[1.0]
print(dataset)


# In[8]:


dataset = dataset[(dataset["State"] == "Kerala") | (dataset["State"] == "Delhi") | (dataset["State"] == "Hubei")]
dataset


# In[10]:

state_len = {}

for state in dataset.State.unique():
   state_len[state] = {
			'population': max(dataset[dataset["State"] == state]["StatePopulation"])/1000000, 
			'days': len(dataset[dataset["State"] == state]["DateI"]), 
			'confirmed_high': max(dataset[dataset["State"] == state]["Confirmed"])
			}
   dataset.loc[dataset["State"] == state, "Confirmed"]=dataset[dataset["State"] == state]["Confirmed"]/state_len[state]["confirmed_high"]
#   dataset.loc[dataset["State"] == state, "DateI"] = dataset[dataset["State"] == state]["DateI"]/state_len[state]["days"]

# In[21]:


def generalized_error_function(t, params) :
    alpha = params[0]
    beta  = params[1]
    p     = params[2]
    return 0.5 * p * ( 1.0 + scipy.special.erf( alpha * ( t - beta ) ) )

# link function used for beta
def identity_fun(x) :
    return x
#
# link function used for alpha, p
def exp_fun(x) :
    return np.exp(x)
#
# inverse of function used for alpha, p
def log_fun(x) :
    return np.log(x)

def return_se_as_f_of_t(t):
    return np.random.normal(scale=0.1)


# In[40]:


model = BasicModel(
    all_data=dataset, 
    col_t="DateI", 
    col_obs="Confirmed", 
    col_group="State",
    col_obs_compare="Confirmed", 
    all_cov_names=[['cov_one'], ['cov_one', 'DaysCovariate'], ["DaysCovariate"]],
    fun=functions.log_erf, 
    predict_space=functions.log_erf, 
    fit_dict={'options': {'disp':False}},  
    basic_model_dict={'col_obs_se': "SE", 
                    'col_covs': [['cov_one'], ['cov_one', 'DaysCovariate'], ["DaysCovariate"]], 
                    'param_names': ['alpha', 'beta', 'p'], 
                    'link_fun': [exp_fun, identity_fun, exp_fun], 
                    'var_link_fun': [exp_fun, identity_fun, exp_fun]}, 
    obs_se_func=return_se_as_f_of_t)


# In[41]:


print("Model pipeline setting up...")
model.setup_pipeline()


# In[43]:


print("Model setup. Running fit...")
model.fit(dataset)


# In[33]:

print(model.mod.params)
model.run(n_draws=180, prediction_times=np.linspace(0, 180, num=180), cv_threshold=0.001, smoothed_radius=[4,4], num_smooths=3, exclude_groups=["Hubei"], exclude_below=20)

print(model.mod.params)
# In[36]:


print("Printing predictions")
predictions_Kerala = model.mean_predictions["Kerala"]
predictions_Maharashtra = model.mean_predictions["Delhi"]
print(predictions_Kerala)
print(predictions_Maharashtra)


# In[38]:





# In[37]:


plt.plot(predictions_Kerala, label="KR")
plt.plot(predictions_Maharashtra, label="UP")
plt.legend(loc="upper left")
plt.show()


# In[ ]:




