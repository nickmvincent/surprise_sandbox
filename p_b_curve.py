from collections import defaultdict, OrderedDict

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


from viz_constants import (
    metric2title, group2scenario, num_users, num_ratings
)

def select_cols(cols, metrics, groups, increase=False, percents=False):
    """
    take a list of cols and filter based on metrics/groups/percents
    
    The returned columns must match all the metrics and groups passed
    if percents is True, return columns with "percent" in the name
    if percents is False return only columns without "percent" in the name
    
    returns a list of cols
    """
    if percents is True and increase is False:
        raise ValueError('Set increase=True to get percent increase')
    ret = []
    for col in cols:
        x = col.split('_')
        
        if percents:
            if 'percent' not in col:
                continue
        else:
            if 'percent' in col:
                continue
        if increase:
            if 'increase' not in col:
                continue
        else:
            if 'increase' in col:
                continue
        for group in groups:
            if 'standards' in group:
                if 'standards' not in col:
                    continue
                #print(group, x)
                group = group.replace('standards_', '')
            else:
                if 'standards' in col:
                    continue
            if group == x[-1]:
                for metric in metrics:
                    if metric == x[-2]:
                        ret.append(col)

    return ret

def fill_in_longform(df):
    """
    Fill in a longform dataframe with metric, group, and name information
    """
    df = df.assign(
        metric=[x.split('_')[-2] for x in df.increase_type]
    )
    df = df.assign(
        group=['standards_' + x.split('_')[-1] if 'standards' in x else x.split('_')[-1] for x in df.increase_type]
    )
    return df

def p_b_curve(
        df, dataset, metrics, groups,
        increase=False, percents=False, normalize=True,
        reg_plot=False, hue='group', row='metric', col='algo_name',
        show_interp=False, legend=True,
        ylabel="", plot_horiz_lines=True, height=5,
        title_template='Recommender {} vs. Size of Boycott for {}',
        aspect=1, palette={'all': 'b', 'non-boycott': 'g', 'standards_non-boycott': 'y'},
        label_map = {
            'all': 'Data \nStrike',
            'non-boycott': 'Data \nStrike +\nBoycott',
            'bizall': 'Data Strike, system',
            'biznon-boycott': 'Boycott, system',
            'standards_non-boycott': 'Trad. Boycott',
        },
        line_names=['all'],
        ylim=None,
        print_vals = [0.1, 0.2, 0.3],
        hue_order=None,
        algo2metric2altalgo=None,
        id_vars=None,
        ds2standards=None,
    ):
    """
    Plots a performance vs. boycott size (P v B) curve
    
    Args:
        df - the dataframe to use for plotting the pb curve
        metrics - which metrics to plot the curve for.
        groups - which groups to include.
          for each metric, 
        percents - show the Y-axis in percent change or raw change. Set this True for percent change.
        normalize - should we normalize the y-axis relative to MovieMean (no personalization)
        reg_plot - ?
        hue - which facet to use to determine hue. default is group
          (each group will appear as a separate (distinctly colored) trajectory on each plot)
        row - which facet to use to split plots into different rows. default is metric.
        save - should we save a PNG file?
        show_interp - include the interpolated function on this plot?
        
    Returns:
        ?
    """
    
    algo_names = list(set(df.algo_name))
    df = df.copy()
    algo2metric2group2 = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    increase_cols = select_cols(list(df.columns.values), metrics, groups, increase=increase, percents=percents)
    #print('increase_cols', increase_cols)
    if normalize:
        for metric in metrics:
            for algo_name in algo_names:
                movie_val = abs(algo2metric2altalgo[algo_name][
                    metric.replace('coverage-weighted-', '')
                ]['MovieMean'])           
                for increase_col in increase_cols:
                    if increase_col.split('_')[-2] == metric:
                        df.loc[df.algo_name == algo_name, increase_col] /= movie_val
    longform = df[increase_cols + id_vars].melt(
        id_vars = id_vars,
        var_name='increase_type'
    )
    longform = fill_in_longform(longform)

    grid = sns.lmplot(
        x="num_users_boycotting", y="value", hue=hue, data=longform,
        sharey='row', sharex='col',
        height=height, 
        aspect=aspect,
        row=row, col=col,
        fit_reg=False,
        x_estimator=np.mean, ci=99, 
        palette=palette,
        legend=legend,
        legend_out=True,
        hue_order=hue_order,
        markers='.',
    )

    algo_to_size_to_decreases = defaultdict(lambda: defaultdict(list))
    
    for metric in metrics:
        for algo_name in algo_names:
            filt = df[df.algo_name == algo_name]
            for group in groups:
                key = '{}_{}'.format(metric, group)
                if increase:
                    key = 'increase_' + key
                if percents:
                    key = 'percent_' + key
                # funky code to handle the fact that we want to treat "standards" as being a unique group
                if 'standards' in key:
                    key = 'standards_' + key.replace('standards_', '')
                x = filt.num_users_boycotting
                user_nums = sorted(list(set(filt.num_users_boycotting)))
                nrm_rounded = sorted(list(set(filt.nrm_rounded)))

                num2mean = OrderedDict()
                roundnum2mean = OrderedDict()
                for num_users_boycotting in user_nums:
                    filt_by_name = filt[filt.num_users_boycotting == num_users_boycotting]
                    num2mean[num_users_boycotting] = np.mean(filt_by_name[key])
                    roundnum2mean[round(num_users_boycotting, 2)] = np.mean(filt_by_name[key])
                nrm_rounded_to_mean = OrderedDict()
                for num in nrm_rounded:
                    filt_by_nrm = filt[filt.nrm_rounded == num]
                    nrm_rounded_to_mean[num] = np.mean(filt_by_nrm[key])

                # for printing out values at various intervals
                if print_vals:
                    if group == 'all':
                        for num in print_vals:
                            val_nb = roundnum2mean.get(num)
                            if val_nb is None:
                                continue
                            val_all = np.mean(filt[filt.num_users_boycotting == num][key.replace('non-boycott', 'all')])
                            print(
                                'Algo:{}  |  Metric:{}  |  #users:{}'.format(
                                    algo_name, metric, num, 
                                )
                            )
                            print('NB Val:{}  |  ALL val:{}'.format(
                                val_nb,
                                val_all
                            ))
                            if percents is False:
                                try:
                                    ratio_nb = val_nb / algo2metric2altalgo[algo_name][metric]['MovieMean']
                                    ratio_all = val_all / algo2metric2altalgo[algo_name][metric]['MovieMean']

                                    print('ratio_nb: {}  |  ratio_all:{}'.format(
                                        ratio_nb,
                                        ratio_all,
                                    ))
                                except KeyError:
                                    print('Metric {} has no MovieMean comparison.'.format(metric))
                            algo_to_size_to_decreases[algo_name][num].append(val_nb)
                meany = np.array(list(num2mean.values()))
                meany_ratings = np.array(list(nrm_rounded_to_mean.values()))

                algo2metric2group2[algo_name][metric][group]['x'] = user_nums
                algo2metric2group2[algo_name][metric][group]['y'] = meany

                smoothf_ratings = interp1d(nrm_rounded, meany_ratings, kind='quadratic', bounds_error=False, fill_value='extrapolate')
                algo2metric2group2[algo_name][metric][group]['interp_ratings'] = smoothf_ratings
                algo2metric2group2[algo_name][metric][group]['max_user_units'] = max(user_nums)
                algo2metric2group2[algo_name][metric][group]['xnew_ratings'] = np.linspace(
                    min(filt.num_ratings_missing), max(filt.num_ratings_missing), num=1000)

    if plot_horiz_lines:        
        for x in grid.facet_data():
            i_row, i_col, i_hue = x[0]
            
            # this is bad hard-coding. Fix after deadline?
            if row == 'metric' and col == 'algo_name' and hue == 'group':
                metric = grid.row_names[i_row]
                algo_name = grid.col_names[i_col]
                group = grid.hue_names[i_hue]
            else:
                group = grid.col_names[i_col]
                _ = grid.row_names[i_row]
                metric = grid.hue_names[i_hue]

            lines_caption = ""

            # we should store all the horiztonal lines in one data structure
            lines = {}

            metric_key = metric.replace('coverage-weighted-', '')
            # needs to handle the coverage-weighted mtrics
            comparisons = algo2metric2altalgo[algo_name][metric_key]
            if increase is False:
                comparisons = ds2standards[dataset]
                comparisons = {
                    key: comparisons[key].get(metric_key, 0) for key in comparisons.keys()
                }
                
            if 'all' in line_names or 'MovieMean' in line_names:
                lines['MovieMean'] = {
                    'value': comparisons['MovieMean'],
                    'color': 'r',
                    'name': 'MovieMean',
                    'linestyle': ':'
                }
            if 'all' in line_names or 'Zero' in line_names:
                lines['Zero'] = {
                    'value': 0,
                    'color': '0.3',
                    'name': 'Zero',
                    'linestyle': ':'
                }
            if 'all' in line_names or 'MaxDamage' in line_names:
                lines['MaxDamage'] = {
                    'value': -114000,
                    'color': '0.3',
                    'name': 'MaxDamage',
                    'linestyle': ':'
                }
                
            if 'GlobalMean' in line_names:
                lines['GlobalMean'] = {
                    'value': comparisons['GlobalMean'],
                    'color': 'y',
                    'name': 'GlobalMean',
                    'linestyle': ':'
                }
            if 'SVD' in line_names:
                lines['SVD'] = {
                    'value': comparisons['SVD'],
                    'color': '0.3',
                    'name': 'SVD',
                    'linestyle': ':'
                }
            if 'KNNBasic_item_msd' in line_names:
                lines['KNNBasic_item_msd'] = {
                    'value': comparisons['KNNBasic_item_msd'],
                    'color': 'c',
                    'name': 'Item KNN (1999)',
                    'linestyle': ':'
                }
            if 'KNNBaseline_item_msd' in line_names:
                lines['KNNBaseline_item_msd'] = {
                    'value': comparisons['KNNBaseline_item_msd'],
                    'color': 'y',
                    'name': 'Item KNN + Baselines: 2010',
                    'linestyle': ':',
                }
            if 'KNNBasic_user_msd' in line_names:
                lines['KNNBasic_user_msd'] = {
                    'value': comparisons['KNNBasic_user_msd'],
                    'color': 'c',
                    'name': 'User KNN (1994)',
                    'linestyle': ':'
                }
            if '1M_SVD' in line_names:
                lines['1M_SVD'] = {
                    'value': comparisons['1M_SVD'],
                    'color': '0.3',
                    'name': 'ml-1m SVD',
                    'linestyle': ':'
                }

            if normalize:
                norm_val = abs(comparisons['MovieMean'])
                for key in lines.keys():
                    lines[key]['value'] /= norm_val

            ax = grid.axes[i_row, i_col]
            
            if ylim:
                ax.set(ylim=ylim)


            #linestyle = '-' if group != 'all' else '--'
            linestyle= '-'
            ax.plot(
                algo2metric2group2[algo_name][metric][group]['x'],
                algo2metric2group2[algo_name][metric][group]['y'],
                linestyle=linestyle, color=grid._colors[i_hue]
            )
            if show_interp:
                xnew_ratings = algo2metric2group2[algo_name][metric][group]['xnew_ratings']
                ynew_ratings = algo2metric2group2[algo_name][metric][group]['interp_ratings'](xnew_ratings)
                ax.plot(
                    xnew_ratings * algo2metric2group2[algo_name][metric][group]['max_user_units'] / num_ratings[dataset],
                    ynew_ratings, '-')

            plt.setp(ax.get_xticklabels(), visible=True, rotation=45)
            
            try:
                metric4title = metric2title[metric]
                ax.set_title(
                    title_template.format(
                        metric4title, dataset.upper())
                )
            except:
                ax.set_title(
                    title_template.format(
                        group2scenario[group], dataset.upper())
                )
                pass
            for line in lines.values():
                ax.axhline(line['value'], color=line['color'], linestyle=line['linestyle'])
                #ax.text(0.7, line['value'] + 0.05, line['name'])
                lines_caption += "{} colored line (value of {}) shows comparison with {}\n".format(
                    line['color'], line['value'], line['name']
                )

    plt.subplots_adjust(hspace=0.2)
    if print_vals:
        algo_to_size_to_mean_dec = defaultdict(dict)
        for algo_name, size2dec in algo_to_size_to_decreases.items():
            for size, decs in size2dec.items():
                algo_to_size_to_mean_dec[algo_name][size] = np.mean(decs)
        print('=====\nSize to Mean Decrease')
        print(algo_to_size_to_mean_dec)
                   
    grid.set_xlabels('Fraction of Users Participating')
    grid.set_ylabels(ylabel)
    if label_map:
        if grid._legend:
            for t in grid._legend.texts:
                t.set_text(label_map[t.get_text()])
            grid._legend.set_title('')
    return algo2metric2group2