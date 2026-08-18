#!/bin/Python3

import pandas as pa
import numpy as np
import sklearn as sk
import statsmodels.api as sm
import growthcurves as gc
import Growth_functions as gf


### Further analysis: What are the growth parameters (maximal growth rate, time to reach plateau, maximal value reached) for each condition 
# +Are there significant differences between duplicates ? Is the fluorescence signal behaving similarly to DO ?

# Fitting the model using growthcurves package (possible models : "mech_logistic","mech_gompertz","mech_richards" & "mech_baranyi")
def model_fitting(measurement_set, time_set, model_type, min_points=10):
    try :
        function_fit = gc.parametric.fit_parametric(time_set, measurement_set, method=model_type, min_points=min_points)
        stat_fit_summary = gc.inference.extract_stats(function_fit, time_set, measurement_set)
    except (ValueError, RuntimeError) as e:
        print(f"  fit échoué ({e}), puits ignoré")
        return None, None
    return function_fit, stat_fit_summary

# Creating array to plot the function fit and the data points
def predicting_values(function_fit,stat_function, time_set):
    if function_fit is None:
        return None,None 
    function_type = function_fit['model_type']
    # Creating an array of time values, the unit is day (0.5 day = 12 h...)
    time_fit = np.linspace(time_set.min(), time_set.max(), 100)
    K = function_fit['params']['K'] 
    mu = function_fit['params']['mu']
    N0 = function_fit['params']['N0']
    if function_type == "mech_baranyi":
        lag = function_fit['params']['h0']/mu
    else:
        lag = stat_function['params']['exp_phase_start']-stat_function['params']['fit_t_min']

    if function_fit is not None:
        if "logistic" in function_type:
            fit_values = gf.logistic_growth(time_fit, N0, K, mu, lag)
        elif "gompertz" in function_type:
            fit_values = gf.gompertz_growth(time_fit, N0, K, mu, lag)
        elif "richards" in function_type:
            fit_values = gf.richards_growth(time_fit, N0, K, mu, lag)
        elif "baranyi" in function_type:
            fit_values = gf.baranyi_growth(time_fit, N0, K, mu, lag)
        else:
            raise ValueError(f"Model type '{function_type}' not recognized.")
    return(np.column_stack((fit_values[0].T, time_fit.T)),fit_values[1])


# Normalizing fluorescence and DO signals values to perform statistical analysis on their behavior (calculation : xnorm = xval/xmax for each well, to keep the same variance and curvatures)

def data_normalization(data_set, measurements_type):
    measure_col = [col for col in data_set.columns if measurements_type in col]
    for col in measure_col:
        data_set[f'Normalized_{col}'] = data_set.groupby(['Row_plate_name', 'Columns_plate_name'])[col].transform(lambda x: (x) / (x.max()))
    return data_set

# Centering and reducing cpnditions data to perform statistical analysis on their behavior using DOE tools (keeping trends and curvatures but removing the absolute values of the measurements)
# Boundaries is a dictionary containing the min and max values for each condition to be centered and reduced, in the form of {'condition_name': [min_value, max_value]}
def data_centering_reduction(condition_set, boundaries):
    conditions = [col for col in condition_set.columns if 'Factor' in col]
    reduced_conditions = condition_set.copy(deep=True)
    for condition in conditions:
        if condition in boundaries:
            lo_val, hi_val = boundaries[condition]
            reduced_conditions[condition] = (2 * (reduced_conditions[condition] - hi_val) / (hi_val - lo_val))+1
        else:
            raise ValueError(f"Boundaries for condition '{condition}' not provided.")
    return reduced_conditions

# Combining the parameters from the growth to extract a value representing desired growth characteristics which will then be used for modelisation according to the conditions
# Combination is as follows : growth_score = [(mumax/mu)*(K-N0)] / [(t_mumax - t_fit_min)/(t_fit_max - t_fit_min)]
def growth_score_evaluation(mumax,mu, K, N0, t_mumax, t_fit_min, t_fit_max):
    w_mu,w_k,w_tmu = 1,1,1
    growth_score = np.log(w_mu*(mumax/mu) + w_k*(K/N0) + w_tmu*((t_mumax ) / (t_fit_max - t_fit_min)))
    return growth_score

# Making a model using scikitlearn polyfeature and ols from statsmodels

def modelisation(df_model, formula='', degree=3, compare=False):
    
    # creating first the X and y variables for the modelisation
    y = df_model['Growth_score']
    X = df_model.drop(columns=['Growth_score'])

    # Getting the feature names for the X variable
    Feature_names = ['constante'] + X.columns.tolist()

    # Adding polynomial features to the X variable plus the constant term for the intercept
    poly = sk.preprocessing.PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_poly = sm.add_constant(X_poly, prepend=True)

    # Creating the model using OLS from statsmodels

    if formula != '':
        model = sm.OLS.from_formula(formula, data=df_model)
    else:
        model = sm.OLS(y, X_poly)

    # Fitting the model
    results = model.fit()

    if compare:
        X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X_poly, y, test_size=0.2, random_state=42)
        model_ml = sk.linear_model.LinearRegression()
        model_ml.fit(X_train, y_train)
        y_pred = model_ml.predict(X_test)
        r2_score = sk.metrics.r2_score(y_test, y_pred)
        rmse_score = np.sqrt(sk.metrics.mean_squared_error(y_test, y_pred))
        print('='*40+'\n')
        print("Model Comparison:\n")
        print('-'*10+'Scikit learn linear regression'+'-'*10+'\n')
        print(f"R² Score: {r2_score:.4f}"+'\n')
        print(f"RMSE: {rmse_score:.4f}"+'\n')
        print(f'Coefficients: {model_ml.coef_}'+'\n')
        #     
        print('\n')
        print('\n')
        #
        print('-'*10+'Statsmodels OLS'+'-'*10+'\n')
        print(f'{results.summary()}'+'\n')
        print('='*40)

    else : 
        print('='*40+'\n')
        print('-'*10+'Statsmodels OLS'+'-'*10+'\n')
        print(f'{results.summary()}'+'\n')
        print('='*40)

    return results, X_poly, poly

# Function to predict growth scores using the significant features from the model and not all the features, to avoid overfitting and to keep only the most relevant features for the prediction
# Unadvised to replace this function by model.predict(X_poly) as it will use all the features and not only the significant ones

def model_prediction(model_results, X_poly, threshold=0.05):

    # Pval associated with the model results
    p_values = np.array(model_results.pvalues[:])

    # Coefficients of the model
    coefficients = model_results.params
    print(f'Coefficients: {coefficients}')

    # Keeping only significant features from the model results
    if len(p_values)==len(coefficients):
        if len(p_values[p_values < threshold]) > 0:
            li_index = [k.tolist()[0] for k in np.where(p_values < threshold)]
            print(li_index)
            coefficients = np.array([coefficients[index] for index in li_index])
            print(f'X_poly: {X_poly}')
            X_poly_significant = X_poly[li_index]
            print(X_poly_significant)
        else:
            print ("No significant features found in the model results. This is highly speculative")
            X_poly_significant=X_poly
    
    # Printing the significant features
    print (f'The following features are significant (pval < {threshold}) in the model: {np.where(p_values < threshold)[0].tolist()}')

    # Predicting the growth score using the significant features and the X_poly_significant
    y_significant_pred = coefficients.T @ X_poly_significant

    return y_significant_pred

# Using the prediction and abscissa values to find maximum values of growth score for lowest concentration of each condition, to find the best growth conditions for the bacteria


    

    
    
