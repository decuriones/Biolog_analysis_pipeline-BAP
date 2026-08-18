#!/usr/bin/env python3

import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import seaborn as sns


def conditions_to_replicate_mapping(conditions_table):
    """
    Build the replicate-mapping table expected by replicate_parsing() (Data_plot.py)
    from a flat per-well conditions table.

    Parameters
    ----------
    conditions_table : pa.DataFrame
        Required columns:
          'row_plate_name' – row letter of the well (e.g. 'A', 'B')
          'col_plate_name' – column number of the well (int or str, e.g. 1 or '10')
          'condition_id'   – any label; wells sharing the same id are replicates

    Returns
    -------
    pa.DataFrame
        'Original_well_ID' : well ID of the first (reference) replicate per condition
        '1 replicate', '2 replicate', ... : additional replicate well IDs
        One row per unique condition_id. Conditions with fewer replicates than
        the maximum are padded with None.

    Example
    -------
    Input:
        row_plate_name  col_plate_name  condition_id
        A               1               1
        B               1               1
        C               1               2
        D               1               2

    Output:
        Original_well_ID  1 replicate
        A1                B1
        C1                D1
    """
    required = {'row_plate_name', 'col_plate_name', 'condition_id'}
    missing = required - set(conditions_table.columns)
    if missing:
        raise ValueError(f"conditions_table is missing required columns: {missing}")

    tbl = conditions_table.copy()
    tbl['_well_id'] = tbl['row_plate_name'].astype(str) + tbl['col_plate_name'].astype(str)

    # Deterministic order: sort within each condition by plate position
    groups = (
        tbl.sort_values(['row_plate_name', 'col_plate_name'])
           .groupby('condition_id', sort=True)['_well_id']
           .apply(list)
    )

    max_n = int(groups.apply(len).max())
    if max_n < 1:
        raise ValueError("conditions_table contains no wells.")

    # N replicates → N columns: 'Original_well_ID', '1 replicate', ..., '{N-1} replicate'
    col_names = ['Original_well_ID'] + [f'{i} replicate' for i in range(1, max_n)]

    rows = [well_ids + [None] * (max_n - len(well_ids))
            for well_ids in groups]

    return pa.DataFrame(rows, columns=col_names).reset_index(drop=True)


def plot_replicates(data_set, measurements_type, save_path=None):
    """
    Plot all replicates of each condition on the same subplot.

    Designed to work with the output of replicate_parsing() from Data_plot.py,
    which adds 'Initial_ID' (condition identifier = Original_well_ID of the
    condition) and 'Replicate_ID' (1-based integer) columns to the dataset.
    One subplot is drawn per condition; each replicate appears as a separate line.
    Multiple measurement columns are distinguished by line style.

    Parameters
    ----------
    data_set : pa.DataFrame
        Must contain 'Initial_ID', 'Replicate_ID', 'Time', and at least one
        column whose name contains measurements_type (case-sensitive).
    measurements_type : str
        Substring used to select measurement columns (e.g. 'Absorbance').
    save_path : str or None
        Directory path. If provided the figure is saved as
        '<save_path>/<measurements_type>_replicates.png' before display.
    """
    required = {'Initial_ID', 'Replicate_ID', 'Time'}
    missing = required - set(data_set.columns)
    if missing:
        raise ValueError(
            f"data_set is missing columns {missing}. "
            "Run replicate_parsing() first to add them."
        )

    measure_col = [col for col in data_set.columns if measurements_type in col]
    if not measure_col:
        raise ValueError(
            f"No column contains '{measurements_type}'. "
            f"Available columns: {list(data_set.columns)}"
        )

    data_set = data_set.copy()
    data_set['Time'] = pa.to_datetime(data_set['Time'], errors='coerce')
    data_set.dropna(subset=['Time'], inplace=True)
    data_set.sort_values(['Initial_ID', 'Replicate_ID', 'Time'], inplace=True)

    conditions = sorted(data_set['Initial_ID'].dropna().unique().tolist(), key=str)
    replicates = sorted(data_set['Replicate_ID'].dropna().unique().tolist())

    if not conditions:
        raise ValueError("No conditions found in 'Initial_ID' column.")

    n_conditions = len(conditions)
    n_cols = min(4, n_conditions)
    n_rows = int(np.ceil(n_conditions / n_cols))

    rep_colors = sns.color_palette("husl", len(replicates))
    rep_color_map = {rep: rep_colors[i] for i, rep in enumerate(replicates)}
    linestyles = ['-', '--', ':', '-.']

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(n_cols * 4.0, n_rows * 3.0),
        squeeze=False,
    )

    for cond_idx, cond_id in enumerate(conditions):
        ax = axes[cond_idx // n_cols, cond_idx % n_cols]
        cond_data = data_set[data_set['Initial_ID'] == cond_id]

        for rep_id in replicates:
            rep_data = cond_data[cond_data['Replicate_ID'] == rep_id]
            if rep_data.empty:
                continue
            for m_idx, measure in enumerate(measure_col):
                label = (
                    f'Rep {int(rep_id)}' if len(measure_col) == 1
                    else f'Rep {int(rep_id)} — {measure}'
                )
                sns.lineplot(
                    data=rep_data,
                    x='Time',
                    y=measure,
                    ax=ax,
                    color=rep_color_map[rep_id],
                    linestyle=linestyles[m_idx % len(linestyles)],
                    linewidth=1.8,
                    marker='o',
                    markersize=3,
                    alpha=0.8,
                    label=label,
                    legend=False,
                )

        ax.set_title(f'Condition: {cond_id}', fontsize=10, fontweight='bold', pad=8)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter('J%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)

    for cond_idx in range(n_conditions, n_rows * n_cols):
        axes[cond_idx // n_cols, cond_idx % n_cols].set_visible(False)

    # Pick legend handles from the first non-empty subplot
    legend_handles, legend_labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            if h:
                legend_handles, legend_labels = h, l
                break
        if legend_handles:
            break

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.00),
            ncol=min(len(legend_handles), 5),
            frameon=True,
            fontsize=9,
            title=f'{measurements_type} — replicates',
            title_fontsize=10,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        plt.savefig(
            f'{save_path}/{measurements_type}_replicates.png',
            bbox_inches='tight',
        )

    plt.show()
