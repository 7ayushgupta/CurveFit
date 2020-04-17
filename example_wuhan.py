#! /bin/python3
# vim: set expandtab:
# -------------------------------------------------------------------------
n_data       = 64
num_params   = 3
alpha_true   = 1
beta_true    = 60
p_true       = 0.05
rel_tol      = 1
# -------------------------------------------------------------------------
import sys
import pandas
import numpy
import scipy
import curvefit
import matplotlib.pyplot as plt



def generalized_gaussian_erf(t, params):
    alpha = params[0]
    beta  = params[1]
    p     = params[2]
    return p/2*(1+scipy.special.erf(alpha*(t-beta)))

def log_generalized_gaussian_erf(t, params):
    return numpy.log(generalized_gaussian_erf(t, params))


def get_plot_predictions(prediction_state):
    n_prediction_length = state_len[prediction_state]['days']
    confirmed_high = state_len[prediction_state]['confirmed_high']
    print('Starting predictions... for state {}'.format(prediction_state))
    print("Original data: ")
    original_data = numpy.exp(dataset[dataset["State"] == prediction_state]["Confirmed"])
    print(original_data)
    print("Prediction data: ")
    predictions = numpy.exp(curve_model.predict(numpy.linspace(0,2,num=2*n_prediction_length), group_name=prediction_state))
    print(predictions) 
    
    plt.plot(numpy.linspace(0,2,num=2*n_prediction_length), predictions*confirmed_high, label="predictions {}".format(prediction_state))
    plt.plot(numpy.linspace(0,1,num=n_prediction_length), original_data*confirmed_high, label="original_data {}".format(prediction_state))
    plt.legend(loc="upper right")
    plt.title("Plots")
    plt.draw()
    plt.pause(0.1)

# model for the mean of the data
def generalized_logistic(t, params) :
    alpha = params[0]
    beta  = params[1]
    p     = params[2]
    # print("Inside logistic fun")
    # print(t)
    return p / ( 1.0 + numpy.exp( - alpha * ( t - beta ) ) )
#
# link function used for beta
def identity_fun(x) :
    return x
#
# link function used for alpha, p
def exp_fun(x) :
    try:
        return numpy.exp(x)
    except:
        print("HAGG DIYA? KYUN HAGG DIYA!")
        print(x)
        pass
#
# params_true
params_true       = numpy.array( [ alpha_true, beta_true, p_true ] )
#
# data_frame
# independent_var   = numpy.array(range(n_data)) * beta_true / (n_data-1)
# measurement_value = generalized_logistic(independent_var, params_true)
# measurement_std   = n_data * [ 0.1 ]
constant_one      = n_data * [ 1.0 ]
# data_group        = n_data * [ 'world' ]
# data_dict         = {
#     'independent_var'   : independent_var   ,
#     'measurement_value' : measurement_value ,
#     'measurement_std'   : measurement_std   ,
#     'constant_one'      : constant_one      ,
#     'data_group'        : data_group        ,
# }
# data_frame        = pandas.DataFrame(data_dict)


dataset = pandas.read_csv('../dataset_for_CurveFit.csv')
dataset = dataset.drop(['Date', 'Unnamed: 0'], axis=1)
state_len = {}

for state in dataset.State.unique():
    state_len[state] = {'population': max(dataset[dataset["State"] == state]["StatePopulation"])/1000000, 'days': len(dataset[dataset["State"] == state]["DateI"]), 'confirmed_high': max(dataset[dataset["State"] == state]["Confirmed"])}
    dataset.loc[dataset["State"] == state, "Confirmed"]=numpy.log(dataset[dataset["State"] == state]["Confirmed"]/state_len[state]['confirmed_high'])
    dataset.loc[dataset["State"] == state, "DateI"] = dataset[dataset["State"] == state]["DateI"]

n_data = dataset.shape[0]
dataset["constant_one"] = n_data * [1.0]
dataset["SE"] = n_data * [0.1]
print("Printing dataset")
print(dataset)
# curve_model
col_t        = 'DateI'
col_obs      = 'Confirmed'
col_covs     = num_params *[ [ 'DaysCovariate' ] ]
col_group    = 'State'
param_names  = [ 'alpha', 'beta',       'p'     ]
link_fun     = [ exp_fun, identity_fun, exp_fun ]
var_link_fun = link_fun
fun          = log_generalized_gaussian_erf
col_obs_se   = 'SE'
#
curve_model = curvefit.CurveModel(
    dataset,
    col_t,
    col_obs,
    col_covs,
    col_group,
    param_names,
    link_fun,
    var_link_fun,
    fun,
    col_obs_se
)

def optimise():
    #    For optimisation the initial values
    fe_init = params_true
    curve_model.fit_params(fe_init)
    params_estimate = curve_model.params
    rel_error = [0,0,0]
    for i in range(num_params) :
        rel_error[i] = params_estimate[i] / params_true[i] - 1.0

    print(rel_error)
    min_error_till_now = numpy.linalg.norm(rel_error)
    min_error_params = params_estimate
    print(min_error_till_now)
    for p in numpy.linspace(0.1, 2, 10):
        for alpha in numpy.linspace(0.1,2,10):
            for beta in numpy.linspace(0.1,2,10):            
                curve_model.fit_params([alpha, beta, p])
                params_estimate = curve_model.params
                #
                for i in range(num_params) :
                    rel_error[i] = params_estimate[i] / params_true[i] - 1.0
                error = numpy.linalg.norm(rel_error)
                if error<min_error_till_now:
                    min_error_till_now = error
                    min_error_params=[alpha,beta,p]
                print("Error occurs at: {}, with error: {}".format([alpha, beta, p], error))

    print("---------------------------------------------")
    print("Minimum error occurs at: {}, with error: {}".format(min_error_params, min_error_till_now))

def estimate(fe_init):
    curve_model.fit_params(fe_init, method="L-BFGS-B", options={'disp':True})
    params_estimate = curve_model.params
    rel_error = [0,0,0]
    for i in range(num_params) :
        rel_error[i] = params_estimate[i] / params_true[i] - 1.0
    error = numpy.linalg.norm(rel_error)
    print(error)
    print('Fitted parameters: OK')

    get_plot_predictions("Uttar Pradesh")
    # get_plot_predictions("Hubei")
    get_plot_predictions("Delhi")

    plt.pause(1000000)

# optimise()
estimate([0.1, 0.1, 0.1])
