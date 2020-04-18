import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from curvefit.core.model import CurveModel
from curvefit.core.functions import ln_gaussian_cdf

np.random.seed(1234)
dataset = pd.read_csv('../dataset_for_CurveFit.csv')
# Create example data -- both death rate and log death rate
df = pd.DataFrame()

df['time'] = np.arange(100)
df['death_rate'] = np.exp(.1 * (df.time - 20)) / (1 + np.exp(.1 * (df.time - 20))) + np.random.normal(0, 0.1, size=100).cumsum()
print(df['death_rate'])
df=pd.DataFrame()
df['time'] = dataset['DateI']
df['death_rate']=dataset['Confirmed']
df['death_rate']=(df['death_rate'].clip(lower=1e-5))*1000000
print(df['death_rate'])

df['ln_death_rate'] = np.log(df['death_rate'])
print(df['ln_death_rate'])
df['group'] = dataset['State']
df['intercept'] = 1.0
print(df)
# Set up the CurveModel
model = CurveModel(
    df=df,
    col_t='time',
    col_obs='ln_death_rate',
    col_group='group',
    col_covs=[['intercept'], ['intercept'], ['intercept']],
    param_names=['alpha', 'beta', 'p'],
    link_fun=[lambda x: x, lambda x: x, lambda x: x],
    var_link_fun=[lambda x: x, lambda x: x, lambda x: x],
    fun=ln_gaussian_cdf
)

# Fit the model to estimate parameters
model.fit_params(fe_init=[0, 0, 1.],
                 fe_gprior=[[0, np.inf], [0, np.inf], [1., np.inf]],
                 options={'disp':True})

# Get predictions
y_pred = model.predict(
    t=np.linspace(0,100,num=100),
    group_name="Kerala"
)
ground_truth = df.ln_death_rate[df['group']=="Hubei"].reset_index(drop=True)
print(ground_truth)
print(np.exp(y_pred))
# Plot results
plt.plot(np.linspace(0,100,num=100), y_pred, '-')
plt.plot(ground_truth, '.')
plt.show()
