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
blanked_data= pa.read_csv(os.path.join(analysis_dir, 'blanked_data.csv'))

# saving preprocessed data
prepared_data= pa.read_csv(os.path.join(analysis_dir, 'preprocessed_data.csv'))

# saving analysis df
Analysis_output= pa.read_csv(os.path.join(analysis_dir, 'growth_parameters.csv'))

# saving normalized analysis df
Normalized_analysis_output= pa.read_csv(os.path.join(analysis_dir, 'normalized_growth_parameters.csv'))

# saving correlation matrices
df_correlation_data= pa.read_csv(os.path.join(analysis_dir, 'correlation_matrices_pre-corr.csv'))

# saving predicted values for each well
prediction_df= pa.read_csv(os.path.join(analysis_dir, 'predicted_values.csv'))
prediction_plot_df= pa.read_csv(os.path.join(analysis_dir, 'predicted_values_2plot.csv'))

# dimensionality reduction
df_pca= pa.read_csv(os.path.join(analysis_dir, 'pca_results.csv'))

# clustering
clusters_linkage_df= pa.read_csv(os.path.join(analysis_dir, 'clusters_linkage.csv'))
df_labeling= pa.read_csv(os.path.join(analysis_dir, 'clusters_HDBSCAN.csv'))

## List variables
List_of_strains= list(blanked_data['strain'].unique())
List_of_elements= list(blanked_data['Element'].unique())
List_of_substrates= list(blanked_data['substrate'].unique())
plates_serial = blanked_data['plate_serial'].unique()
Assays = pa.Series(plates_serial).str.extract(r"^(?P<assay>.+)-\d+$")["assay"].unique()
List_measurements = [measure for measure in blanked_data.columns if 'measurement' in measure.lower() or 'absorbance' in measure.lower() or 'fluorescence' in measure.lower() or 'luminescence' in measure.lower()]


### Plotting functions

# Plotting growth curves for each assay and serial plate

# I had too much issue with this one and took what claude opus5 gave me, but I fully understand it
def mean_replicate_curves(db, measure, x_col='delta_time', n_points=100):
    """
    Courbe moyenne des réplicats d'un assay, puits par puits.
 
    Retourne un DataFrame long : Row_plate_name, Columns_plate_name, x_col,
    '{measure}_mean', '{measure}_sd', 'n_replicates'.
 
    La grille est bornée à [max des t_min, min des t_max] sur les réplicats :
    np.interp EXTRAPOLE en répétant la valeur de bord, ce qui fabriquerait un
    plateau artificiel là où un réplicat s'est arrêté plus tôt.
    """
    sub = db.loc[db['plate_serial'].str.startswith(assay)]
    if sub.empty:
        return pa.DataFrame()
 
    bornes = sub.groupby('plate_serial')[x_col].agg(['min', 'max'])
    t_min, t_max = bornes['min'].max(), bornes['max'].min()
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        print(f'  {assay} : pas de fenêtre temporelle commune, moyenne ignorée')
        return pa.DataFrame()
    grid = np.linspace(t_min, t_max, n_points)
 
    morceaux = []
    for (r, c), well in sub.groupby(['Row_plate_name', 'Columns_plate_name']):
        courbes = []
        for _, rep in well.groupby('plate_serial'):
            rep = rep.dropna(subset=[x_col, measure]).sort_values(x_col)
            if len(rep) < 2:
                continue
            courbes.append(np.interp(grid, rep[x_col].to_numpy(float),
                                     rep[measure].to_numpy(float)))
        if not courbes:
            continue
        arr = np.vstack(courbes)
        morceaux.append(pa.DataFrame({
            'Row_plate_name': r,
            'Columns_plate_name': c,
            x_col: grid,
            f'{measure}_mean': arr.mean(axis=0),
            f'{measure}_sd': arr.std(axis=0, ddof=1) if len(arr) > 1 else 0.0,
            'n_replicates': len(arr),
        }))
        return pa.concat(morceaux, ignore_index=True) if morceaux else pa.DataFrame()
def plot_growth_curves(db, db_pred, output_dir, assays, plates_serial,
                       list_measurements, measurements_type='absorbance',
                       pattern='', obs_x='delta_time', pred_x='Time_fit_plot',
                       pred_value='Prediction_2plot'):
    """
    Pour chaque assay :
      - une figure 'moyenne des réplicats' (moyenne épaisse + réplicats fins)
      - pour chacune de SES plaques : données brutes, puis fit + observations
    """
    for ass in assays:
        path_assay = os.path.join(output_dir, ass)
        os.makedirs(path_assay, exist_ok=True)
 
        # plaques de CET assay uniquement (sinon 18x le travail, mauvais dossier)
        plaques_ass = [ps for ps in plates_serial if str(ps).startswith(ass)]
        if not plaques_ass:
            continue
        db_ass = db.loc[db['plate_serial'].str.startswith(ass)]
 
        # ---------------------------------------------- moyenne des réplicats
        for measure in list_measurements:
            db_mean = mean_replicate_curves(db_ass, ass, measure, x_col=obs_x)
            if db_mean.empty:
                continue
 
            # 1) la moyenne, épaisse, au premier plan
            fig_mean, ax_mean = dp.data_plotting(
                db_mean, measurements_type, save_path=None, type='line',
                x_col=obs_x, value_col=f'{measure}_mean',
                color='tab:red', label='moyenne', zorder=10)
 
            # 2) chaque réplicat par-dessous, une couleur par plaque
            couleurs = sns.color_palette('Blues_d', len(plaques_ass))
            for i, serial in enumerate(plaques_ass):
                rep = db_ass.loc[db_ass['plate_serial'] == serial]
                if rep.empty:
                    continue
                dp.data_plotting(rep, measurements_type, save_path=None,
                                 type='scatter', x_col=obs_x, value_col=measure,
                                 ax=ax_mean, style_axes=False,
                                 color=couleurs[i], label=str(serial),
                                 zorder=3 + i)
 
            for lg in list(fig_mean.legends):
                lg.remove()
            handles = [plt.Line2D([], [], color='tab:red', lw=2.5, label='mean')]
            handles += [plt.Line2D([], [], color=couleurs[i], lw=1.2, label=str(s))
                        for i, s in enumerate(plaques_ass)]
            fig_mean.legend(handles=handles, loc='upper center',
                            bbox_to_anchor=(0.5, 1.0),
                            ncol=min(len(handles), 5), frameon=True, fontsize=9)
            for a in ax_mean.flat:
                if a.get_ylabel():
                    a.set_ylabel(measure, fontsize=8)
 
            fig_mean.savefig(
                os.path.join(path_assay,
                             f'growth_curves_mean_{ass}_{measure}_{pattern}.png'),
                dpi=200, bbox_inches='tight')
            plt.close(fig_mean)          # sinon avertissement après 20 figures
    return None

##

def plot_growth_curves_pred(db, db_pred, output_dir, measurements_type='absorbance', pattern = ''):
    for ass in Assays:
        for ps in plates_serial:
            os.makedirs(os.path.join(path_assay, ps), exist_ok=True)

            # simple plot of actual data
            dp.data_plotting (db.loc[db['plate_serial'] == ps], measurements_type,os.path.join(path_assay, ps), filename=f'growth_curves_{ps}_{pattern}', type='line')

            # plot of predicted data, with actual data in the background

            fig, ax = dp.data_plotting(db_pred.loc[db_pred['plate_serial'] == ps], measurements_type, save_path=None,
                            type='line', x_col=pred_x, value_col=pred_value,
                            color='tab:red', label='fit', zorder=2)
 
            dp.data_plotting(db.loc[db['plate_serial'] == ps], measurements_type, save_path=None,
                  type='scatter', x_col=obs_x, ax=ax,
                  style_axes=False, color='tab:blue', label='observed', zorder=3)
            filename=f'growth_curves_predicted&observed_{ps}_{pattern}.png'

            # Legends tend to overlap on each other so we remove them and add a global legend at the end of the plotting function
            for lg in list(fig.legends):
                lg.remove()
            handles = [plt.Line2D([], [], color='tab:red', lw=2, label='fit_prediction'),
                        plt.Line2D([], [], color='tab:blue', marker='o', ls='',
                                    label='observed')]
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                        ncol=2, frameon=True, fontsize=10)
            
            for a in ax.flat:
                if a.get_ylabel():
                    a.set_ylabel(measurements_type, fontsize=8)
 
            if output_dir:
                fig.savefig(os.path.join(path_assay, ps, filename),
                            dpi=200, bbox_inches='tight')
    return(fig, ax) # fig and ax are only reprensentative of the last plot

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

## Heatmap plot for area under the curve (AUC) values

def AUC_heatmap_plot(analysis, save_dir=None, value_col='AUC',
                     col_key='assay', annot=False, zscore_rows=False):
    """
    Une heatmap par couple (Element, measurement).
 
    analysis      : Analysis_output ou Normalized_analysis_output
    value_col     : 'AUC' ou 'Normalized_AUC'
    col_key       : 'assay' (réplicats moyennés) ou 'strain'
    zscore_rows   : centre-réduit chaque ligne -> met en avant les contrastes
                    entre assays plutôt que le niveau absolu du substrat
    Retourne un dict {(Element, measurement): (fig, ax)}.
    """
    auc_df = analysis.copy()
 
    if value_col not in auc_df.columns:
        raise ValueError(f"'{value_col}' absent. col : {list(auc_df.columns)}")
 
    # assay attribution
    auc_df['assay'] = auc_df['plate_serial'].str.extract(r"^(?P<assay>.+)-\d+$")["assay"]
 
    auc_df = auc_df.dropna(subset=[value_col, col_key])
    if auc_df.empty:
        raise ValueError(f"All line are missing {value_col}")
 
    # Unambiguous id attribution
    auc_df['substrate_id'] = auc_df['substrate'] + '_' + auc_df['Element']
 
    outputs = {}
    # a graph per (Element, measurement) couple
    for (element, measurement), bloc in auc_df.groupby(['Element', 'measurement']):
 
        # pivot_table aggregate replicate (mean) and deal with duplication
        mat = bloc.pivot_table(index='substrate_id', columns=col_key,
                               values=value_col, aggfunc='mean')
        if mat.empty or mat.shape[1] < 1:
            continue
 
        title = f'{value_col}_{element}_{measurement}-Heatmap'
        if zscore_rows:
            # standard deviation nul (1 col) -> let 0 instead of NaN
            sd = mat.std(axis=1).replace(0, np.nan)
            mat = mat.sub(mat.mean(axis=1), axis=0).div(sd, axis=0).fillna(0) # calculating z-score per row (mean over standard deviation)
            title += ' (z-score per row)'
 
        # height proportional to number of rows, width proportional to number of columns
        fig, ax = plt.subplots(
            figsize=(max(4, 1.1 * mat.shape[1] + 2), max(6, 0.22 * mat.shape[0])))
        sns.heatmap(mat, cmap='vlag' if zscore_rows else 'viridis',
                    center=0 if zscore_rows else None,
                    annot=annot, fmt='.2f',
                    linewidths=0.3, linecolor='white',
                    cbar_kws={'label': value_col}, ax=ax)
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_xlabel(col_key)
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
 
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(
                save_dir, f'{value_col}_heatmap_{element}_{measurement}.png'),
                dpi=200, bbox_inches='tight')
        outputs[(element, measurement)] = (fig, ax)
 
    return outputs

def whisker_plot_AUC(analysis, save_dir=None, value_col='AUC', col_key='assay'): ### Allow to roughly compare proeficiency of a given strain for a given element
    """
    Un boxplot par couple (Element, measurement).
 
    analysis      : Analysis_output ou Normalized_analysis_output
    value_col     : 'AUC' ou 'Normalized_AUC'
    col_key       : 'assay' (réplicats) ou 'strain'
    Retourne un dict {(Element, measurement): (fig, ax)}.
    """
    auc_df = analysis.copy()
 
    if value_col not in auc_df.columns:
        raise ValueError(f"'{value_col}' absent. col : {list(auc_df.columns)}")
 
    # assay attribution
    auc_df['assay'] = auc_df['plate_serial'].str.extract(r"^(?P<assay>.+)-\d+$")["assay"]
 
    auc_df = auc_df.dropna(subset=[value_col, col_key])
    if auc_df.empty:
        raise ValueError(f"All line are missing {value_col}")
 
    # Unambiguous id attribution
    auc_df['substrate_id'] = auc_df['substrate'] + '_' + auc_df['Element']
 
    outputs = {}
    # a graph per (Element, measurement) couple
    for (element, measurement), bloc in auc_df.groupby(['Element', 'measurement']):
        fig, ax = plt.subplots(figsize=(max(4, 1.1 * len(bloc[col_key].unique()) + 2), 6))
        sns.boxplot(data=bloc, x=col_key, y=value_col, ax=ax)
        ax.set_title(f'{value_col}_{element}_{measurement}-Boxplot', fontsize=12, pad=12)
        ax.set_xlabel(col_key)
        ax.set_ylabel(value_col)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
 
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(
                save_dir, f'{value_col}_boxplot_{element}_{measurement}.png'),
                dpi=200, bbox_inches='tight')
        outputs[(element, measurement)] = (fig, ax)
 
    return outputs


### Execution of the plotting functions

for meas in List_measurements:
    # Plotting growth curves for each assay and serial plate
    plot_growth_curves(blanked_data, prediction_plot_df, fig_dir, Assays, plates_serial, List_measurements, measurements_type=meas, pattern='{meas}', obs_x='delta_time', pred_x='Time_fit_plot', pred_value='Prediction_2plot')

    # Plotting growth curves for each assay and serial plate with predicted values
    plot_growth_curves_pred(blanked_data, prediction_plot_df, fig_dir, measurements_type=meas, pattern = '{meas}')

# Plotting the correlation matrices
plot_correlation_matrices(df_correlation_data, fig_dir)

# Plotting PCA results
plot_pca(df_pca, features=List_measurements)

# Plotting heatmaps for AUC values
AUC_heatmap_plot(Analysis_output, save_dir=fig_dir, value_col='AUC', col_key='assay', annot=False, zscore_rows=False)
AUC_heatmap_plot(Normalized_analysis_output, save_dir=fig_dir, value_col='Normalized_AUC', col_key='assay', annot=False, zscore_rows=False)

# Plotting boxplots for AUC values
whisker_plot_AUC(Analysis_output, save_dir=fig_dir, value_col='AUC', col_key='assay')
whisker_plot_AUC(Normalized_analysis_output, save_dir=fig_dir, value_col='Normalized_AUC', col_key='assay')