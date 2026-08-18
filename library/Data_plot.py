#!/bin/Python3

from matplotlib.dates import DateFormatter
import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import copy
import seaborn as sns
import os


# utilities :

def counting(list,pattern):
    counting = 0
    for element in list:
        if pattern in element:
            counting += 1
    return counting

### Loading data and extracting metadata

def load_data(file_path):
    # Quickly opening the dataset and extract the line at which data starts
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Finding the line at which each features starts (e.g. "Experiment details", "Procedure details", "Layout", "Temperature") and the last line of the temperature section to know where the data starts
    procedure_idx = max(i for i, line in enumerate(lines) if "Procedure Details" in line)
    layout_idx = max(i for i, line in enumerate(lines) if "Layout" in line)
    results_idx = max(i for i, line in enumerate(lines) if "Results" in line)
    data_idx = max(i for i, line in enumerate(lines) if "Actual Temperature" in line)
    
    parsing = {
            'Experiment details': lines[:procedure_idx],
            'Procedure details': lines[procedure_idx+1:layout_idx],
            'Layout': lines[layout_idx:results_idx],
            'Temperature': lines[results_idx+1:data_idx]
        }
    # Creating metadata dictionary with empty values for each key
    metadata = {}
    for k in parsing.keys():
        metadata[k] = {}
        
    # Filling main metadata values
    for key in ['Experiment details', 'Temperature']:
        for cat in parsing[key]:
            cat = list(filter(None, cat.strip().split(';')))
            if len(cat) > 1:
                if cat[0] in metadata[key].keys():
                    cat[0]=f'{cat[0]}_{counting(metadata[key].keys(), cat[0])}'
                metadata[key][cat[0]]=cat[1]

    # Procedure parsing to metadata ready for use
    for Procedure in parsing['Procedure details']:
        Procedure = list(filter(None, Procedure.strip().split(';')))
        if len(Procedure) > 1:
            if Procedure[0] in metadata['Procedure details'].keys():
                    
                Procedure[0]=f'{Procedure[0]}_{counting(metadata["Procedure details"].keys(), Procedure[0])}'
            metadata['Procedure details'][Procedure[0]]=[Procedure[1]]
            Last_procedure = Procedure[0]
        elif len(Procedure)==1:
            metadata['Procedure details'][Last_procedure].append(Procedure[0])
        
    # Layout parsing to metadata ready for use
    for Layout in parsing['Layout'][2:]:
        Parsed_layout = list(filter(None, Layout.strip().split(';')))
        for index in range(len(Parsed_layout[1:])):
            metadata['Layout'][f'{Parsed_layout[0]}_{index}']=Parsed_layout[index]

    # opening dataset
    data_set = pa.read_csv(file_path, sep=';', skiprows=data_idx+2, encoding='utf-8')
    data_set.drop(columns='Unnamed: 0', inplace=True, errors='ignore')
    # extracting metadata
    with open(file_path, 'r') as f:
        file = f.readlines()[:53]
        
        
    return data_set, metadata



### Modifying data to correspond pandas dataframe formatting and adding time column (from metadata)

def data_formating(data_set, metadata):

    col_unnamed = [col for col in data_set.columns if col.startswith('Unnamed')]
    # Changing columns names to correspond to pandas dataframe formatting
    data_set.rename({col_unnamed[0]: 'Row_index', col_unnamed[-1]: 'Measurement'}, inplace=True, axis='columns')

    # Dropping "Blanked" measurements rows automatically added by omnilog software
    Blank_index = data_set[data_set['Measurement'].str.startswith('Blank')].index
    data_set.drop(Blank_index, inplace=True)

    # Modifying every Nan to the corresponding row name in the row index column (equivalent to extending a value in excel)
    Last_row_name = np.nan
    for idx in data_set.index:
        if not pa.isna(data_set.loc[idx, 'Row_index']):
            Last_row_name = data_set.loc[idx, 'Row_index']
        else:
            data_set.loc[idx, 'Row_index'] = Last_row_name

    # Rotation the dataframe such as columns representing plate col number become rows and that each measurement is a column, excluding Row_index column
    rotation_array = np.concatenate([np.array(["Row_plate_name", "Columns_plate_name"]), np.array(data_set.loc[data_set['Row_index']==data_set['Row_index'].unique()[0],data_set.columns[-1]].values)],axis=0).reshape(1,-1)
    for row_name in data_set['Row_index'].unique():
        # Extracting the values corresponding to the row name and creating an array with the row name and the corresponding column numbers to be able to concatenate it to the tmp_array and then rotate it
        tmp_array = np.array(data_set.loc[data_set['Row_index']==row_name,data_set.columns[1:len(data_set.columns)-1]].values)
        # creating and array with the row name and the corresponding column numbers to be able to concatenate it to the tmp_array and then rotate it
        index_array = np.concatenate((np.full((1,tmp_array.shape[1]), row_name),np.arange(1,tmp_array.shape[1]+1,1).reshape(1,-1)), axis=0)
        tmp_array = np.concatenate((index_array, tmp_array), axis=0)
        tmp_array = np.rot90(tmp_array)
        rotation_array = np.concatenate((rotation_array, tmp_array), axis=0)
    rotated_df = pa.DataFrame(rotation_array, columns=rotation_array[0])
    # to understand the rotation, look at the columns naming ! (but basically, a tmp array is created with the two first rows being the row name and the column number, then the values corresponding to the row name are added to it, then the array is rotated and concatenated to the final array)
    # Note that everything is done sequentially, to keep the order, the first row A is processed first, then B, etc. and the final array is created by concatenating the tmp arrays for each row name.

    # Fluorescence col are just nammed "***,***" replacing * by integer, so this pose an issue for later and require correct renaming of all similar col
    col_names = list(rotated_df.columns)
    for col in col_names:
        if col.count(',') == 1 and col.replace(',', '').isdigit():
            new_col = f'fluorescence_measurment_(excitation-{col.split(",")[0]}_emmission-{col.split(",")[1]})'
            rotated_df.rename(columns={col: new_col}, inplace=True)

    # Modifying column names to better suits pandas dataframe formatting
    rotated_df.columns = [f'{col.split(" ")[-2]} {col.split(" ")[-1]}'
                           if len(col.split(" "))>1 else col
                            for col in rotated_df.columns]
    
    
    # Adding time column to dataset
    # Source files store dates in day-first format (dd/mm/yyyy).
    time = pa.to_datetime(
        f"{metadata['Experiment details']['Date']} {metadata['Experiment details']['Time']}",
        dayfirst=True,
        errors='coerce'
    )
    rotated_df['Time'] = time

    # Modifying column names to correspond pandas dataframe formatting
    data_set.columns = [col.strip() for col in data_set.columns]

    # Droping the first row of the dataset which is now useless after rotation, this first row formerly contained the col names
    rotated_df.drop(0, inplace=True)

    # Replacing ',' by '.' in the dataset to be able to convert values to numeric and plot them
    for col in rotated_df.columns:
        if not col in ['Time','Row_plate_name']:
            rotated_df[col] = rotated_df[col].replace(',','.', regex=True)
            rotated_df[col] = pa.to_numeric(rotated_df[col], errors='coerce')

    return rotated_df

### Formatting conditions mapping to correspond pandas dataframe formatting (col = [row_plate_name,col_plate_name,condition1, condition2, condition3])
def fomatting_conditions_mapping(mapping_df):
    formatted_df = pa.DataFrame()
    col_names = mapping_df.columns[:]
    row_names = mapping_df[col_names[0]].unique()
    for row in row_names:
        for col in col_names[1:]:
            formatted_df = pa.concat((formatted_df, pa.DataFrame([[row, col, mapping_df.loc[mapping_df[col_names[0]]==row, col].values[0]]], columns=['Row_plate_name', 'Columns_plate_name', 'Condition'])), ignore_index=True)
    return formatted_df

### Gathering data from evey files at different time points

def data_gathering(folder_path):
    data_list = []
    for file in os.listdir(folder_path):
        print(f'Processing file : {os.path.basename(file)}')
        if file.endswith('.csv'):
            data_set, metadata = load_data(os.path.join(folder_path, file))
            data_list.append(data_formating(data_set, metadata))
    return pa.concat(data_list)

### Finding the blank wells with a key word in their name (e.g. "Blank") and returning a list of their well ID (e.g. A1, B2, etc.)

def blank_wells_finder(mapping_df, keyword):
    blank_wells = []
    for row in mapping_df['Row_plate_name'].unique():
        for col in mapping_df['Columns_plate_name'].unique():
            if type(mapping_df.loc[(mapping_df['Row_plate_name']==row) & (mapping_df['Columns_plate_name']==col), 'Condition'].values[0]) == str and keyword in mapping_df.loc[(mapping_df['Row_plate_name']==row) & (mapping_df['Columns_plate_name']==col), 'Condition'].values[0]:
                blank_wells.append(f'{row}{col}')
    return blank_wells

### Blanking the data according to given wells

def data_blanking(data_set, blank_wells, mode='wells'):

    data_set_blanked = data_set.copy(deep=True)
    measure_cols = [c for c in data_set.columns
                    if 'measurement' in c.lower() or 'absorbance' in c.lower() or 'fluorescence' in c.lower() or 'luminescence' in c.lower()]

    # Blanking the data according to the blank wells given in the list of blank wells, by subtracting the mean of the blank wells values at each time point to the corresponding values of each well at the same time point
    if mode == 'wells':
        for time in data_set['Time'].unique():
            blank_list = pa.DataFrame()
            for well in blank_wells:
                blank_list = pa.concat((blank_list, data_set.loc[(data_set['Row_plate_name']==well[0]) & (data_set['Columns_plate_name']==int(well[1])) & (data_set['Time']==time), :]), ignore_index=True)
            blank_data = blank_list.mean(numeric_only=True)
            for col in measure_cols:
                    data_set_blanked.loc[data_set_blanked['Time']==time, col] = data_set_blanked.loc[data_set_blanked['Time']==time, col].astype(float) - blank_data[col].astype(float)
        return data_set_blanked

    # Blanking the data according to the blank values given in the list of blank values, by subtracting the mean of the blank values at each time point to the corresponding values of each well at the same time point
    elif mode == 'values':
        for time in data_set['Time'].unique():
            blank_data = pa.DataFrame(blank_wells).mean(numeric_only=True)
            for col in measure_cols:
                    data_set_blanked.loc[data_set_blanked['Time']==time, col] = data_set_blanked.loc[data_set_blanked['Time']==time, col].astype(float) - blank_data[col].astype(float)
        return data_set_blanked

    # Blanking the data according to the initial values of each well, by subtracting the initial value of each well to the corresponding values of each well at each time point
    elif mode == 'Initial_values':
        for well in data_set['Row_plate_name'].unique():
            for col in data_set['Columns_plate_name'].unique():
                well_mask = ((data_set['Row_plate_name'] == well)
                         & (data_set['Columns_plate_name'] == col))
                initial_row = data_set.loc[(data_set['Row_plate_name']==well) & (data_set['Columns_plate_name']==col) & (data_set['Time']==data_set['Time'].min()), measure_cols]
                initial_value = initial_row.iloc[0]
                data_set_blanked.loc[well_mask, measure_cols] = (
                    data_set_blanked.loc[well_mask, measure_cols].astype(float)
                                    .sub(initial_value, axis='columns')
                )
        return data_set_blanked

### Plotting every relevant wells given value according to time

def data_plotting(data_set, measurements_type, save_path, filename='Omnilog_data',
                                    type='line', ax=None, value_col=None, x_col='delta_time',
                                    style_axes=True, color=None, label=None, zorder=2):
    
    data_set = data_set.copy()
 
    # Extracting columns corresponding to measurement type wanted
    if value_col is not None:
        measure_col = [value_col]
    else:
        measure_col = [c for c in data_set.columns if measurements_type in c]
    if not measure_col:
        raise ValueError(f"No columns found for measurement type '{measurements_type}' in the dataset.")


    
    ### Sorting dataset according to plate plan and time
    data_set.sort_values(by=['Row_plate_name', 'Columns_plate_name', x_col],
                         inplace=True)

    ### Plotting every relevant wells given measurement type wanted, according to time in a multi-plot figure
    
    # Extracting wells names for plotting
    wells_line = sorted(data_set['Row_plate_name'].unique().tolist())
    wells_col = sorted(data_set['Columns_plate_name'].unique().tolist())

    ## Plot management 
    n_rows, n_cols = len(wells_line), len(wells_col)
    
    # Sizing: plus de subplots = plus grand

    subplot_width = 3.5   # Width per subplot
    subplot_height = 2.5  # Height per subplot
    figsize = (n_cols * subplot_width, n_rows * subplot_height)

    # Define colors for different measurements
    if color is None:
        palette = sns.color_palette("husl", len(measure_col))
        color_map = {m: palette[i] for i, m in enumerate(measure_col)}
    else:
        color_map = {m: color for m in measure_col}
 

    # Creating multi-plot figure with subplots corresponding to each well and plotting the corresponding values according to time for each measurement type wanted
    if ax is None:
        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols,
                               figsize=figsize)
    else:
        fig = ax.flat[0].get_figure()
    
    print(measure_col)
    for row_idx, line in enumerate(wells_line):
        for col_idx, col in enumerate(wells_col):
            current_ax = ax[row_idx, col_idx]
            # Empty wells are just ignored and replaced by "Empty" mention
            well = data_set.loc[(data_set['Row_plate_name'] == line)
                                & (data_set['Columns_plate_name'] == col)]
            if well[measure_col].isna().all().all() or len(well)==0:
                if style_axes:
                    print(f'Well {line} {col} is empty, deleting corresponding subplot')
                    current_ax.text(0.5, 0.5, 'Empty', 
                                ha='center', va='center', 
                                fontsize=10, color='gray', alpha=0.5)
                    current_ax.set_xticks([])
                    current_ax.set_yticks([])
                    current_ax.spines['top'].set_visible(False)
                    current_ax.spines['right'].set_visible(False)
                    current_ax.spines['left'].set_visible(False)
                    current_ax.spines['bottom'].set_visible(False)
                continue

            # Actual plotting of the values according to time for each measurement type wanted in the corresponding subplot
            for measure in measure_col:
                if type == 'line':
                    sns.lineplot(data=well,
                                x=x_col, y=measure,
                                label=label if label is not None else measure,
                                ax=current_ax,
                                color=color_map[measure],
                                linewidth=2,
                                marker='o',
                                markersize=4,
                                alpha=0.8,
                                legend=False,
                                zorder=zorder,
                                squeeze=False)
                elif type == 'scatter':
                    sns.scatterplot(data=well,
                                    x=x_col, y=measure,
                                    label=label if label is not None else measure,
                                    ax=current_ax,
                                    color=color_map[measure],
                                    s=30,
                                    alpha=0.8,
                                    legend=False, 
                                    zorder=zorder,
                                    squeeze=False)
                if style_axes:
                    # ============ SUBPLOT STYLING ============
                    # Title with well ID
                    current_ax.set_title(f'{line}{col}', fontsize=11, fontweight='bold', pad=10)
                    current_ax.set_xlabel('jours', fontsize=8)

                    # Grid for readability
                    current_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    current_ax.set_axisbelow(True)
                        
                    # Styling
                        
                    current_ax.spines['top'].set_visible(False)
                    current_ax.spines['right'].set_visible(False)
                    current_ax.tick_params(labelsize=8)
    if style_axes:
        # ============ GLOBAL LEGEND ============
        # Create legend outside the subplots
        handles, labels = ax[0, 0].get_legend_handles_labels()
        if handles:  # Only if there are lines to legend
            fig.legend(
                handles, 
                labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.00),  # Above the figure
                ncol=len(measure_col),
                frameon=True,
                fontsize=10,
                title=f'Measurement Type: {measurements_type}',
                title_fontsize=11
            )

        # ============ LAYOUT ============
        if style_axes:
            fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for global legend

        # ============ ADD GRID LINES to subplots ============
        # Make subplot borders visible
        for axes in ax.flat:
            for spine in axes.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)
                spine.set_visible(True)
    
        
    # Saving figure in the corresponding folder
    if save_path:
        fig.savefig(f'{save_path}/{filename}_{measurements_type}_plot.png')
    
    return fig, ax  # Return the figure and axes for further manipulation if needed
    


# Function to plot duplicates of the same conditions in the same subplot

def plot_duplicates(data_set, measurements_type, conditions_map):
    return()


# Parsing mapping according to template (e.g. similar to the one used in the usual data_formating function) with columns representing the plates columns and rows representing the plates lines

def mapping_parsing(mapping_df, metadata):
    
    # Rotation the dataframe such as columns representing plate col number become rows and that each measurement is a column, excluding Row_index column
    rotation_array = np.concatenate([np.array(["Row_plate_name", "Columns_plate_name"]), np.array(mapping_df.loc[mapping_df['Row/Col ID']==mapping_df['Row/Col ID'].unique()[0],mapping_df.columns[1:14]].values)],axis=0).reshape(1,-1)
    for row_name in mapping_df['Row_index'].unique():
        tmp_array = np.array(mapping_df.loc[mapping_df['Row_index']==row_name,mapping_df.columns[1:14]].values)
        index_array = np.concatenate((np.full((1,tmp_array.shape[1]), row_name),np.arange(1,tmp_array.shape[1]+1,1).reshape(1,-1)), axis=0)
        tmp_array = np.concatenate((index_array, tmp_array), axis=0)
        tmp_array = np.rot90(tmp_array)
        rotation_array = np.concatenate((rotation_array, tmp_array), axis=0)
    rotated_df = pa.DataFrame(rotation_array, columns=rotation_array[0])

    # Modifying column names to better suits pandas dataframe formatting
    rotated_df.columns = [f'{col.split(" ")[-2]} {col.split(" ")[-1]}'
                           if len(col.split(" "))>1 else col
                            for col in rotated_df.columns]
    
    
    # Adding time column to dataset
    # Source files store dates in day-first format (dd/mm/yyyy).
    time = pa.to_datetime(
        f"{metadata['Experiment details']['Date']} {metadata['Experiment details']['Time']}",
        dayfirst=True,
        errors='coerce'
    )
    rotated_df['Time'] = time

    # Modifying column names to correspond pandas dataframe formatting
    mapping_df.columns = [col.strip() for col in mapping_df.columns]

    # Droping the first row of the dataset which is now useless after rotation
    rotated_df.drop(0, inplace=True)

    return rotated_df

# Function to parse duplicate an associate them to a same well with a replicate ID and original well ID, to be able to plot them in the same subplot and perform statistical analysis on them
def replicate_parsing(data_set, replicate_mapping):
    
    Parsed_duplicates = copy.deepcopy(data_set)

    # Taking a the dataset and a dataframe that map in a line each well ID associated to a given conditions and in columns (col name 'n replicate') the well ID of the n replicate (e.g. 3 replicates for each condition, thus 3 columns '1 replicate', '2 replicate' and '3 replicate' in the mapping dataframe) 
    # Note that well ID must be consitently written as follow: 'Row_plate_name' + 'Columns_plate_name' (e.g. A1, B2, etc.)
    if len(replicate_mapping.columns) < 2:
        raise ValueError("The replicate mapping dataframe must have at least two columns: one for the original well ID and at least one for the replicates.")
    else :
        for well_id in replicate_mapping['Original_well_ID'].unique():
            if well_id not in Parsed_duplicates['Row_plate_name'].agg(lambda x: str(x)) + Parsed_duplicates['Columns_plate_name'].agg(lambda x: str(x)):
                raise ValueError(f"Original well ID {well_id} not found in the dataset.")
            else:
                line = str(well_id[0])
                col = int(well_id[1:])
                replicate = 1
                for replicate_col in replicate_mapping.columns:
                    Parsed_duplicates.loc[(Parsed_duplicates['Row_plate_name']==line) & (Parsed_duplicates['Columns_plate_name']==col), 'Initial_ID'] = replicate_mapping.loc[replicate_mapping['Original_well_ID']==well_id, replicate_col].values[0]
                    Parsed_duplicates.loc[(Parsed_duplicates['Row_plate_name']==line) & (Parsed_duplicates['Columns_plate_name']==col), 'Replicate_ID'] = replicate
                    Parsed_duplicates.loc[(Parsed_duplicates['Replicate_ID']==replicate) & (Parsed_duplicates['Initial_ID']==replicate_mapping.loc[replicate_mapping['Original_well_ID']==well_id, replicate_col].values[0]), ['Row_plate_name','Columns_plate_name']] = well_id[0],int(well_id[1:])
                    replicate += 1
    return Parsed_duplicates
