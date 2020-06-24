import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def missing_report(df, pct_threshold=0):
    missing_vals_n = df.isna().sum().sort_values(ascending=False)
    missing_vals_pct = round(missing_vals_n / df.shape[0], 3)

    missing_vals = pd.concat([missing_vals_n, missing_vals_pct], axis=1, keys=['n', 'pct'])

    return missing_vals[missing_vals['pct'] > pct_threshold]


def handle_missing_values(df, drop_vars):
    df_clean = df.copy()
    df_clean.drop(drop_vars, inplace=True, axis=1)
    df_clean.dropna(inplace=True)

    return df_clean


def outlier_report(df, pct_threshold=0):
    df_numeric = df.select_dtypes(['int', 'float']).copy()
    outliers = df_numeric.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3)  # .all(axis=1).value_counts()

    n_outliers = pd.melt(outliers.apply(pd.Series.value_counts).iloc[[0]])
    n_outliers['pct_outliers'] = round(n_outliers['value'] / df.shape[0], 2)
    n_outliers_ordered = n_outliers.dropna().sort_values('value', ascending=False)

    return n_outliers_ordered[n_outliers_ordered['pct_outliers'] > pct_threshold]


def handle_outlier_values(df, drop_vars):
    """Detect outliers that are 3 std away from mean (1% highest and lowest)"""
    df_clean = df.select_dtypes(['int', 'float']).copy()
    df_clean.drop(drop_vars, inplace=True, axis=1)
    detect_outliers = df_clean.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)

    df.drop(drop_vars, inplace=True, axis=1)
    print(detect_outliers.value_counts())

    return df[detect_outliers]


def normalize(df):
    result = df.copy()
    for feature_name in df.select_dtypes(['int', 'float']).columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def plt_feature_distribution(df, chart_height, save_fig=False, output_path=None):
    order = pd.melt(df.select_dtypes(['int', 'float'])).groupby('variable')['value'].mean().sort_values()[::-1].index

    fig = plt.figure(figsize=(14, chart_height))
    sns.boxplot(x="value", y="variable", data=pd.melt(df.select_dtypes(['int', 'float'])), order=order, fliersize=0.5)

    plt.xlabel('')
    plt.ylabel('')
    plt.title('Feature Set Distribution')

    if save_fig is True:
        if output_path is None:
            raise ValueError('Need to specify output path to save the plot')
        else:
            plt.savefig(output_path, bbox_inches='tight')

    return plt.show()


def feature_delta_by_target(df, target_var, metric, top_n, drop_vars=None):
    df = df.groupby(target_var).describe()

    target_breakdown = df.xs(metric, level=1, axis=1).transpose()
    target_breakdown['delta'] = np.where(target_breakdown[1] - target_breakdown[0] == 0, 0,
                                         ((target_breakdown[1] - target_breakdown[0]) / target_breakdown[0]))

    target_breakdown['delta_abs'] = abs(target_breakdown['delta'])
    target_breakdown.replace([-np.inf, np.inf], 0, inplace=True)
    target_breakdown.sort_values('delta_abs', ascending=False, inplace=True)
    target_breakdown.drop('delta_abs', axis=1, inplace=True)
    target_breakdown = target_breakdown.loc[
        ~target_breakdown['delta'].isin([0, 1])].copy()  # Remove fields with high sub-group cardinality.

    if drop_vars is None:
        pass
    else:
        target_breakdown.drop(drop_vars, inplace=True)

    return target_breakdown.head(top_n).round(3).style.background_gradient(cmap='viridis', subset='delta')


def plt_feature_dist_by_target(df, fields, target_var, save_fig=False, output_path=None):
    # df_melt = pd.melt(df.drop(fields, axis=1).select_dtypes(['int', 'float']), target_var)
    df_melt = pd.melt(df[fields].select_dtypes(['int', 'float']), target_var)

    g = sns.FacetGrid(df_melt, col='variable', hue=target_var, col_wrap=4, aspect=1.5, palette='tab10', sharey=False,
                      sharex=False, legend_out=False)
    g = g.map(sns.kdeplot, 'value', shade=True)

    g.set_titles('{col_name}', fontsize=4)
    g.set_yticklabels('')
    g.set_xlabels('')

    g.add_legend()
    g.fig.tight_layout()

    if save_fig is True:
        if output_path is None:
            raise ValueError('Need to specify output path to save the plot')
        else:
            plt.savefig(output_path, bbox_inches='tight')

    return plt.show()


def bivariate_plot(df, x_var, y_var, x_label, y_label, save_fig=False, output_path=None):
    from matplotlib.ticker import FormatStrFormatter

    sns.set_style(style="white")
    gridkw = dict(height_ratios=[4, 3])

    if df[x_var].dtype in ('int', 'float'):
        x, edges = pd.cut(df[x_var], 10, retbins=True)
    else:
        raise ValueError('Categorical variable currently not supported')

    f, ax = plt.subplots(2, sharex=False, figsize=(14, 8), gridspec_kw=gridkw, constrained_layout=True)

    sns.regplot(x=df[x_var], y=df[y_var], order=3, scatter=False, x_ci=0.05, ax=ax[0])  # x_estimator=np.mean,
    sns.barplot(x=x, y=df[y_var], color='steelblue', ci=None, ax=ax[1])  # df[x_var].round(1)

    #     for p in ax[1].patches:
    #         ax[1].annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
    #                        ha='center', va='center', fontsize=11, color='gray', xytext=(0, 10), textcoords='offset points')

    ax[0].set_xticks([])
    ax[0].set_xlabel('')
    ax[0].set_ylabel(y_label)
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel('Volumes')
    ax[1].set_xticklabels(edges.round(1))
    f.suptitle(x_label + ' vs ' + y_label)

    if save_fig is True:
        if output_path is None:
            raise ValueError('Need to specify output path to save the plot')
        else:
            plt.savefig('Outputs/biplot_' + output_path, bbox_inches='tight')

    return plt.show()


def minimal_bar(series, ax=None, width=0.8, fisize=(6, 3),
                reorder_yaxis=True, splines_off=True, delete_ticks=True, y_label_large=True, display_value=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=fisize)

    # 1. Delete legend legend=False
    # 2. Tighten the space between bars width=0.8
    series.plot(kind='barh', legend=False, ax=ax, width=width, color='C0');

    # 3. Re-order the y-axis
    if reorder_yaxis:
        ax.invert_yaxis()

    # 4. Delete the square spines
    if splines_off:
        [spine.set_visible(False) for spine in ax.spines.values()]

    # 5. Delete ticks for x and y axis
    # 6. Delete tick label for x axis
    if delete_ticks:
        ax.tick_params(bottom=False, left=False, labelbottom=False)

    # 7. Increase the size of the label for y axis
    if y_label_large:
        ax.tick_params(axis='y', labelsize='x-large')

    # 8. Display each value next to the bar
    if display_value:
        vmax = series.max()
        for i, value in enumerate(series):
            ax.text(value + vmax * 0.02, i, f'{value:,}', fontsize='x-large', va='center', color='C0')
