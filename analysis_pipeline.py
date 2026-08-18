### Importation step
# Importation of commonly used libraries
import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Importation of special libraries for data analysis
from scipy.integrate import trapezoid
from scipy import stats as st
from statsmodels.stats.runs import runstest_1samp
from sklearn.cluster import HDBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler





# Appending the path to the python files for data processing, formatting, and visualization to the system path to be able to import them
import sys
import os
dir_path = os.path.abspath("/home/elouanln/projects/def-jcomte/elouanln/Sandbox/Code/Phenotyping/Omnilog/Analysis/library")
if dir_path not in sys.path:
    sys.path.append(dir_path)

# Importation of python files for data processing, formatting, and visualization
import Data_plot as dp
import Growth_analysis as da

### Data importation step

# Importation from DB omnilog
path_to_data = ""
output_dir = "/home/elouanln/scratch/Sandbox/DB_omnilog/"
analysis_dir = os.path.join(output_dir, "analysis_results")
os.makedirs(analysis_dir, exist_ok=True)


data_collection = pa.read_csv(path_to_data)

### Data input to adapt analysis pipeline to the data structure
# smoothing data
moving_median = False
FLOOR = 1e-5

# setting the model type and the minimum number of points to be used for the fitting step
model_type = "mech_baranyi"
min_points = 10

# initializing the list of rows, columns and measurements to be analyzed
List_row = data_collection['Row_plate_name'].unique()
List_col = data_collection['Columns_plate_name'].unique()
List_measurements = [measure for measure in data_collection.columns if 'measurement' in measure.lower() or 'absorbance' in measure.lower() or 'fluorescence' in measure.lower() or 'luminescence' in measure.lower()]

# Extracting all assays (counting triplicates) and plates serial
plates_serial = data_collection['plate_serial'].unique()
Assays = pa.Series(plates_serial).str.extract(r"^(?P<assay>.+)-\d+$")["assay"].unique()

### All preprocessing steps
blanked_data = data_collection.copy(deep=True)
#Sorting df for better handling of the data
blanked_data.sort_values(['plate_serial', 'Row_plate_name', 'Columns_plate_name', 'Time'], inplace=True)

## Blanking step
# Can be values, wells or initial values
mode = 'Initial_values'
List_blank = []
if mode == 'wells':
    List_blank=[]
elif mode == 'values':
    List_blank=[]

for serial in plates_serial:
    blanked_data[blanked_data['plate_serial'] == serial] = dp.data_blanking(blanked_data[blanked_data['plate_serial'] == serial], List_blank, mode=mode)

# Note that later fit requires values striclty positive, so if blanking is done with initial values, the data will be shifted to be strictly positive
blanked_data[blanked_data[List_measurements].lt(0).any(axis=1)] = 1/1000000000000000000

## Creating a columns for delta time between each measurement and the first measurement, which will be used for the calculation of the area under the curve (AUC) and the growth rate
for serial in plates_serial:
    # Converting the time set to datetime format and normalizing it to start at 0 for each well
    time_set = pa.to_datetime(blanked_data.loc[blanked_data['plate_serial'] == serial,'Time'])  # Converting the time set to datetime format
    time_set = (time_set - time_set.min()).dt.total_seconds()/3600/24  # Normalizing the time set to start at 0 for each well
    blanked_data.loc[blanked_data['plate_serial'] == serial, 'delta_time'] = time_set

## Note that this part is facultative as the reasoning introduce bias that can heavily affect the results of the analysis
## Smoothing step - data are noisy due to colonial phenotypes of cyanobacterial growth and cyanobacteria also seems to display a circadian rhythm in their growth, which can be seen in the growth curves. 
# This circadian rhythm is not of interest for the analysis and can be smoothed out by applying a moving median locally to the data.
prepared_data = blanked_data.copy(deep=True)
prepared_data[List_measurements] = prepared_data[List_measurements].clip(lower=FLOOR) # Thresholding minimum values to 1e-4 as the fitting step would reject completely too low points
prepared_data.sort_values(['plate_serial', 'Row_plate_name', 'Columns_plate_name', 'Time'], inplace=True)


# hampel filter set outliers to local median. Outlier are defined as k*MAD (median absolute deviation) away from the local median.
def hampel(df, window=7, k=3.0):
    med = df.rolling(window, center=True, min_periods=1).median()
    dev = (df - med).abs()
    mad = dev.rolling(window, center=True, min_periods=1).median()
    sigma = 1.4826 * mad
    return df.mask((sigma > 0) & (dev > k * sigma), med)

if moving_median:
    for serial in plates_serial:
        for measurement in List_measurements:
            prepared_data.loc[prepared_data['plate_serial'] == serial, measurement] = hampel(prepared_data.loc[prepared_data['plate_serial'] == serial, measurement], window=7, k=3.0)

### Analysis step

# initializing the dataframe to collect all growth parameters for desired wells and the dataframe to collect the predicted values for each well
Analysis_output = pa.DataFrame()
prediction_df = pa.DataFrame()


for serial in plates_serial:
    for row in List_row:
        for col in List_col:
            well_mask = (prepared_data['plate_serial'] == serial) & (prepared_data['Row_plate_name'] == row) & (prepared_data['Columns_plate_name'] == col)
            well_data_for_fit = prepared_data.loc[well_mask, :]
            well_data_for_integration = blanked_data.loc[well_mask, :]
            if well_data_for_fit.empty or well_data_for_integration.empty:
                print(f'Well {row}{col} : No data available')
                continue
            strain = well_data_for_fit['strain'].values[0]
            substrate = well_data_for_fit['substrate'].values[0]
            for measurement in List_measurements:
                
                ## Area under the curve (AUC) calculation
                # this methods is not biased by the heterogeneity of measurements frequency
                AUC = trapezoid(well_data_for_integration[measurement], well_data_for_integration['delta_time'])

                ## Model fitting step
                # Adding wells growth parameters to the dictionary
                time_set = well_data_for_fit['delta_time'].values
                measurement_set = well_data_for_fit[measurement].values
                growth_parameters, fitting_statistics = da.model_fitting( measurement_set, time_set, model_type, min_points)
                
                if growth_parameters is None or fitting_statistics is None:
                    print(f'Well {row}{col} : No growth parameters or fitting statistics could be extracted for measurement {measurement}')
                    continue
                # Parameter attribution from dict growth_parameters to variables for better readability
                # Note that time was defined in delta DAYS so every derivated parameters are related to days
                mu = growth_parameters['params']['mu']
                k = growth_parameters['params']['K']
                N0 = growth_parameters['params']['N0']
                h0 = growth_parameters['params']['h0']
                t_min = growth_parameters['params']['fit_t_min']
                t_max = growth_parameters['params']['fit_t_max']
                model = growth_parameters['model_type']

                # Parameter attribution from dict fitting_statistics to variables for better readability
                max_od = fitting_statistics['max_od']
                mu_max = fitting_statistics['mu_max']
                t_mumax = fitting_statistics['time_at_umax']
                od_at_umax = fitting_statistics['od_at_umax']
                doubling_time = fitting_statistics['doubling_time']
                exp_phase_start = fitting_statistics['exp_phase_start']
                exp_phase_end = fitting_statistics['exp_phase_end']
                model_rmse = fitting_statistics['model_rmse']

                Analysis_output = pa.concat([Analysis_output, pa.DataFrame({'plate_serial': [serial], 'Row_plate_name': [row], 'Columns_plate_name': [col], 'measurement': [measurement], 'strain': [strain], 'substrate': [substrate], 
                'mu': [mu], 'K': [k], 'N0': [N0], 'h0': [h0], 'fit_t_min': [t_min], 'fit_t_max': [t_max], 'model_type': [model_type], 
                'max_od': [max_od], 'mu_max': [mu_max], 'time_at_umax': [t_mumax], 'od_at_umax': [od_at_umax], 'doubling_time': [doubling_time], 
                'exp_phase_start': [exp_phase_start], 'exp_phase_end': [exp_phase_end], 'model_rmse': [model_rmse], 'AUC': [AUC]})], ignore_index=True)


                print("="*20+'\n')
                print(Analysis_output.loc[(Analysis_output['plate_serial'] == serial) & (Analysis_output['Row_plate_name'] == row) & (Analysis_output['Columns_plate_name'] == col) & (Analysis_output['measurement'] == measurement), :])

                # Creating a dataframe with the predicted values for each well and adding it to the final dataframe
                pred, pred2plot, time_fit_plot = da.predicting_values(growth_parameters, time_set) # using growth curve existing function for prediction
                if pred is None:
                    print(f'Well {row}{col} : No predicted values could be extracted')
                    continue


                # prediction df and plotting df
                prediction_df = pa.concat([prediction_df, pa.DataFrame({'Plate_name': serial, 'Row_plate_name': row, 'Columns_plate_name': col, f'Predicted_values_{measurement}': pred, 'Time': time_set})], ignore_index=True,axis=0)
                prediction_plot_df = pa.concat([prediction_plot_df, pa.DataFrame({'Plate_name': serial, 'Row_plate_name': row, 'Columns_plate_name': col, f'Predicted_values_2plot_{measurement}': pred2plot, 'Time_fit_plot': time_fit_plot})], ignore_index=True,axis=0)

                ## Exploiting prediction to assess statistical significance of the fit
                # simplification of variables for better readability
                y = measurement_set
                y_pred = pred

                # F-test for the significance of the fit
                n = len(y)
                n_params = len(growth_parameters['params']-2) # substracting 1 for the intercept and 1 also for the model type parameter, which is not a parameter 
                ss_null = np.sum((y - y.mean())**2)
                ss_model = np.sum((y - y_pred)**2)
                F = ((ss_null - ss_model) / (n_params - 1)) / (ss_model / (n - n_params))
                p = st.f.sf(F, n_params - 1, n - n_params)

                # Runs test for randomness of residuals
                z, pz = runstest_1samp(y - y_pred, cutoff=0)

                Analysis_output.loc[(Analysis_output['plate_serial'] == serial) & (Analysis_output['Row_plate_name'] == row) & (Analysis_output['Columns_plate_name'] == col) & (Analysis_output['measurement'] == measurement), 'F_test_pvalue', 'Residual_randomness', 'Residual_randomness_pvalue'] = p,z,pz
        
## Creating a correlation matrix for the growth parameters and statistics using pearson and spearman correlation coefficients
df_correlation_data = Analysis_output.copy(deep=True)
df_correlation_data = df_correlation_data[['mu', 'K', 'N0', 'h0', 'fit_t_min', 'fit_t_max', 'max_od', 'mu_max', 'time_at_umax', 'od_at_umax', 'doubling_time', 'exp_phase_start', 'exp_phase_end']]


## Plotting the correlation matrices !!!! à déplacer dans le fichier dédié au plottttttt
for method in ('pearson', 'spearman'):
    corr = df_correlation_data.corr(method=method, numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))          # nouvelle figure à chaque fois
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
                annot=True, fmt='.2f', cmap='vlag', center=0,
                vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title(f'Corrélation {method}')
    fig.savefig(os.path.join(FIG_DIR, f'correlation_matrix_{method}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)  

## Managing parameters too correlated for the next dimensionality reduction step
List_correlated_parameters = []
List_correlated_parameters_log = []

features = [feat for feat in Analysis_output.columns if feature not in List_correlated_parameters and feature not in List_correlated_parameters_log]
# note that feature log should be logged not simply removed, I am just to tired for now ...
df_feat = Analysis_output[features].copy(deep=True)
df_feat = df_feat.dropna()

# Dealing with replicates by taking median values for each condition 
mat = (df_feat.groupby(['strain', 'substrate'])[features]
              .median().reset_index())

## Dimensionality reduction step using PCA to reduce the number of parameters to be used for clustering

X = mat[features].dropna()
Robscal = RobustScaler()
Xs = Robscal.fit_transform(X)
pca = PCA()
scores = pca.fit_transform(Xs)
loadings = pa.DataFrame(pca.components_[:2].T, columns=['PC1', 'PC2'], index=features)

# Printing the explained variance ratio and the loadings for the first two principal components, ULTRA important for interpretation and qc
print(pca.explained_variance_ratio_.round(3))
print(loadings.round(2))

## The plot should be relocated in a py file dedicated to plotting for better readability and modularity of the code
df_pca = mat.loc[X.index].copy()
df_pca['PC1'], df_pca['PC2'] = scores[:, 0], scores[:, 1]

fig, ax = plt.subplots(figsize=(7, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='strain', s=60, ax=ax)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})')
fig.savefig('pca.png', dpi=300, bbox_inches='tight')
plt.close(fig)

## Clustering of the wells based on their growth parameters
df_clustering = Analysis_output.copy(deep=True)
df_clustering = df_clustering.dropna(subset=['mu', 'K', 'N0', 'h0', 'fit_t_min', 'fit_t_max', 'max_od', 'mu_max', 'time_at_umax', 'od_at_umax', 'doubling_time', 'exp_phase_start', 'exp_phase_end'],inplace=True)
# Keeping indexes aligned
df_labeling = df_clustering.copy(deep=True)

df_clustering = df_clustering[['mu', 'K', 'N0', 'h0', 'fit_t_min', 'fit_t_max', 'max_od', 'mu_max', 'time_at_umax', 'od_at_umax', 'doubling_time', 'exp_phase_start', 'exp_phase_end']]
df_labeling = df_labeling[['plate_serial', 'Row_plate_name', 'Columns_plate_name', 'measurement']]

# Clustering using HDBSCAN 
hdb = HDBSCAN(copy=True, min_cluster_size=5)
hdb.fit_predict (df_clustering.values)

# Recording the cluster labels in the original dataframe
df_labeling['cluster_HDBSCAN'] = hdb.labels_

# Clustering using linkage
results = []

for strain, sub in mat.groupby('strain'):
    sub = sub.copy()
    Xs = RobustScaler().fit_transform(sub[features])
    Z = linkage(Xs, method='ward')
    sils = {k: silhouette_score(Xs, fcluster(Z, k, 'maxclust')) for k in range(2, 9)}
    k_opt = max(sils, key=sils.get)
    sub['cluster'] = fcluster(Z, k_opt, 'maxclust')
    sub['silhouette'] = sils[k_opt]
    results.append(sub)
    pic = sns.clustermap(pa.DataFrame(Xs, columns=features, index=sub['substrate']),
                   row_linkage=Z, cmap='vlag')
    pic.savefig(f'os.path.join(analysis_dir, f"clustermap_{strain}.png")', dpi=300, bbox_inches='tight')
    plt.close(pic.figure)

    
clusters_linkage_df = pa.concat(results, ignore_index=True)

### Saving all analysis results to csv files for further analysis and visualization

# saving blanked data
blanked_data.to_csv(os.path.join(analysis_dir, 'blanked_data.csv'), index=False)

# saving preprocessed data
prepared_data.to_csv(os.path.join(analysis_dir, 'preprocessed_data.csv'), index=False)

# saving analysis df
Analysis_output.to_csv(os.path.join(analysis_dir, 'growth_parameters.csv'), index=False)

# saving correlation matrices
df_correlation_data.to_csv(os.path.join(analysis_dir, 'correlation_matrices_pre-corr.csv'), index=False)

# saving predicted values for each well
prediction_df.to_csv(os.path.join(analysis_dir, 'predicted_values.csv'), index=False)
prediction_plot_df.to_csv(os.path.join(analysis_dir, 'predicted_values_2plot.csv'), index=False)

# dimensionality reduction
df_pca.to_csv(os.path.join(analysis_dir, 'pca_results.csv'), index=False)

# clustering
clusters_linkage_df.to_csv(os.path.join(analysis_dir, 'clusters_linkage.csv'), index=False)
df_labeling.to_csv(os.path.join(analysis_dir, 'clusters_HDBSCAN.csv'), index=False)





