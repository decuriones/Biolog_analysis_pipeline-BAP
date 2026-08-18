### Importation step
# Importation of commonly used libraries
import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

# Appending the path to the python files for data processing, formatting, and visualization to the system path to be able to import them
import sys
import os
dir_path = os.path.abspath("/home/elouanln/projects/def-jcomte/elouanln/Sandbox/Code/Phenotyping/Omnilog/Analysis/library")
if dir_path not in sys.path:
    sys.path.append(dir_path)

# Importation of python files for data processing, formatting, and visualization
import Data_plot as dp
import Growth_analysis as da

### importing all dataframe from the analysis pipeline directory

# path variables
output_dir = "/home/elouanln/scratch/Sandbox/DB_omnilog/"
analysis_dir = os.path.join(output_dir, "analysis_results")
fig_dir = os.path.join(output_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# importing the analysis output dataframe
blanked_data= pa.read_csv(os.path.join(analysis_dir, 'blanked_data.csv'), index_col=0)

# saving preprocessed data
prepared_data= pa.read_csv(os.path.join(analysis_dir, 'preprocessed_data.csv'), index_col=0)

# saving analysis df
Analysis_output= pa.read_csv(os.path.join(analysis_dir, 'growth_parameters.csv'), index_col=0)

# saving normalized analysis df
Normalized_analysis_output= pa.read_csv(os.path.join(analysis_dir, 'normalized_growth_parameters.csv'), index_col=0)

# saving correlation matrices
df_correlation_data= pa.read_csv(os.path.join(analysis_dir, 'correlation_matrices_pre-corr.csv'), index_col=0)

# saving predicted values for each well
prediction_df= pa.read_csv(os.path.join(analysis_dir, 'predicted_values.csv'), index_col=0)
prediction_plot_df= pa.read_csv(os.path.join(analysis_dir, 'predicted_values_2plot.csv'), index_col=0)

# dimensionality reduction
df_pca= pa.read_csv(os.path.join(analysis_dir, 'pca_results.csv'), index_col=0)

# clustering
clusters_linkage_df= pa.read_csv(os.path.join(analysis_dir, 'clusters_linkage.csv'), index_col=0)
df_labeling= pa.read_csv(os.path.join(analysis_dir, 'clusters_HDBSCAN.csv'), index_col=0)

## List variables
List_of_strains= list(blanked_data['strain'].unique())
List_of_elements= list(blanked_data['Element'].unique())
List_of_substrates= list(blanked_data['substrate'].unique())
plates_serial = blanked_data['plate_serial'].unique()
Assays = pa.Series(plates_serial).str.extract(r"^(?P<assay>.+)-\d+$")["assay"].unique()


### Plotting functions

# Plotting growth curves for each assay and serial plate
def plot_growth_curves(db, db_pred, output_dir, measurements_type='absorbance'):
    for ass in Assays:
        path_assay= os.path.join(output_dir, ass)
        os.makedirs(path_assay, exist_ok=True)


        for ps in plates_serial:
            os.makedirs(os.path.join(path_assay, ps), exist_ok=True)

            # simple plot of actual data
            fig_actual_data, ax_actual_data= dp.data_plotting (db.loc[db['plate_serial'] == ps], measurements_type,'', filename=f'growth_curves_{ps}', type='line')
            fig_actual_data.savefig(os.path.join(path_assay, ps, f'growth_curves_{ps}.png'), dpi=300, bbox_inches='tight')

            # plot of predicted data, with actual data in the background
            fig_pred, ax_pred = dp.data_plotting (db_pred.loc[db_pred['plate_serial'] == ps], measurements_type, os.path.join(path_assay, ps), filename=f'predicted_growth_curves_{ps}', type='line')
            


            
    return()

## Plotting the correlation matrices !!!! à déplacer dans le fichier dédié au plottttttt ## A ne pas prendre en compte pour la correction
def plot_correlation_matrices(df_correlation_data, FIG_DIR):
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
        return fig, ax # fig and ax are only reprensentative of the last plot
  
## PCA plot
## The plot should be relocated in a py file dedicated to plotting for better readability and modularity of the code
def plot_pca(mat, features):
    df_pca = mat.loc[X.index].copy()
    df_pca['PC1'], df_pca['PC2'] = scores[:, 0], scores[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='strain', s=60, ax=ax)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})')
    fig.savefig('pca.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax