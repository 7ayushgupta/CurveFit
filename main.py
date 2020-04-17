import curvefit
import pandas as pd
import pickle
import datetime
import numpy as np
from curvefit.model_generators import ModelPipeline, BasicModel
from curvefit.functions import expit, log_erf, identity, exponential
from curvefit.pv import PVModel
import matplotlib.pyplot as plt

def return_se_as_f_of_t(t):
    return np.random.normal(scale=0.1)

dataset = pd.read_csv('dataset_for_CurveFit.csv')
dataset = dataset.reset_index(drop=True)
dataset = dataset.sort_values(by=["DateI"])
dataset["SE"] = np.random.normal(scale=0.1, size=dataset.shape[0])
print(dataset["State/UnionTerritory"])

model = BasicModel(all_data=dataset, col_t="DateI", col_obs="Confirmed", col_group="State/UnionTerritory",
                 col_obs_compare="Confirmed", all_cov_names=["DaysCovariate", "DaysCovariate", "DaysCovariate"], fun=log_erf, predict_space=log_erf, fit_dict={'fe_init': [1, 60, 0.05]},  basic_model_dict={'col_obs_se': "SE", 'col_covs': [["DaysCovariate"], ["DaysCovariate"], ["DaysCovariate"]], 'param_names': ['alphalink', 'betalink', 'gammalink'], 'link_fun': [exponential, identity, exponential], 'var_link_fun': [exponential, identity, exponential]}, obs_se_func=return_se_as_f_of_t)

print("Model pipeline setting up...")
model.setup_pipeline()
print("Model setup. Running fit...")
model.fit(dataset)
print("Model fitted. Saving model...")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved.")

model.run(n_draws=180, prediction_times=np.linspace(0, 180, num=180), cv_threshold=0.001, smoothed_radius=[4,4], exclude_groups=["Hubei"], exclude_below=20)
#model.plot_draws(np.linspace(50, 70, num=20), sharex=True, sharey=False)
# plt.show()
predictions_Kerala = np.exp(model.mean_predictions["Kerala"])
predictions_Maharashtra = np.exp(model.mean_predictions["Maharashtra"])
print(predictions_Kerala)
print(predictions_Maharashtra)
plt.plot(np.exp(predictions_Kerala), label="KR")
plt.plot(np.exp(predictions_Maharashtra), label="MH")
plt.legend(loc="upper left")
plt.show()