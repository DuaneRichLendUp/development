import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates

from sklearn.datasets import make_classification

def make_groups_classification(n_samples, n_feats, n_informative,
                               n_redundant, n_classes, random_state,
                               n_individuals, max_count_cat, target_name,
                               group_name, date_name=None,
                               min_date=datetime.datetime(2010, 1, 1),
                               max_date=datetime.datetime(2017, 1, 1)):
    features, targets = make_classification(n_samples=n_samples,
                                            n_features=n_feats,
                                            n_informative=n_informative,
                                            n_redundant=n_redundant,
                                            n_classes=n_classes,
                                            random_state=random_state)
    df = pd.DataFrame(features, columns=['F' + str(f) for f in range(n_feats)])

    # Change some of the feature to categorical

    letters = [chr(i) for i in range(97, 97 + 26)]

    for f in range(n_feats // 2, n_feats):
        num_cats = np.random.choice(range(2, max_count_cat), 1)[0]
        cats = [''.join(np.random.choice(letters, 3)) for i in range(num_cats)]
        min_val = min(df.iloc[:, f].values)
        max_val = max(df.iloc[:, f].values)
        diff = (max_val - min_val) / num_cats
        splits = [min_val + diff * i for i in range(num_cats + 1)]
        cat_vals = [cats[0]] * df.shape[0]
        for i in range(1, num_cats):
            cat_vals = [(cats[i] if df.iloc[j, f] > splits[i] else cat_vals[j]) for j in range(df.shape[0])]
        df.iloc[:, f] = cat_vals

    df[target_name] = targets

    # make the groups
    unique_group_ids = [''.join(np.random.choice(letters, 3)).upper() for i in range(n_individuals)]
    df[group_name] = np.random.choice(unique_group_ids, df.shape[0])

    # make the dates
    if date_name:

        date_diff = (max_date - min_date).days

        df[date_name] = min_date

        for g_id in unique_group_ids:
            logi = [e == g_id for e in df[group_name]]
            user_count = np.sum(logi)
            if user_count >= date_diff:
                rand_days = np.asarray(range(user_count))
            else:
                rand_days = np.random.choice(range(date_diff), user_count, replace=False)
            user_dates = [min_date + datetime.timedelta(days=rd) for rd in rand_days]
            df.loc[logi, date_name] = user_dates

    return df


def get_TS_graphs(df, target_name, group_name, date_name, n_samples=2, figsize=(20, 17)):
    """
    This function generates target-coded timeseries graphs for sampled groups. It's
    purpose is to get a general idea of a group's behavior and how that behavior
    might relate to the target. Specifically, given a dataframe (df) with a classification
    target (the column target_name), this function samples n_samples groups
    (according to the 'group_name' column). Then for each sampled group, a single column of graphs
    is made. Each row of that column indicates one feature of that group. If it's a
    categorical feature, a scatter plot is made, where a particular y-level is associated
    with a particular categorical value. If it's a real valued feature, its a line plot.
    The x-axis is the date (indicated by the 'date_name' column).

    """

    real_cols = df.select_dtypes(include=[np.number]).columns.values
    cat_cols = df.select_dtypes(include=[object, 'category'])

    def exclude(lst):
        return list(set(lst) - set([target_name, group_name, date_name]))

    real_cols = exclude(real_cols)
    cat_cols = exclude(cat_cols)

    gb = df.groupby(group_name)

    sampled_groups = np.random.choice(list(gb.groups.keys()), n_samples)

    # First, we extract the feature ranges and unique categorical values for the samples.
    # We do *not* make sure the samples have the same date ranges.
    feature_ranges = {}
    cat_uniques = {}
    first = True
    for gn in sampled_groups:
        g = gb.get_group(gn)
        if first:

            for rc in real_cols:
                feature_ranges[rc] = [min(g[rc]), max(g[rc])]

            for cc in cat_cols:
                cat_uniques[cc] = list(np.unique(g[cc]))

            first = False

        else:

            for rc in real_cols:
                feature_ranges[rc] = [min(min(g[rc]), feature_ranges[rc][0]),
                                      max(max(g[rc]), feature_ranges[rc][1])]

            for cc in cat_cols:
                cat_uniques[cc] = list(set(cat_uniques[cc] + list(np.unique(g[cc]))))

    def gen_cat_mapper(list_str):
        out = {}
        for i, s in enumerate(list_str):
            out[s] = i + 0.5
        return out

    # Make the graphs
    n_feats = len(real_cols + cat_cols)
    fig, axarr = plt.subplots(n_feats, n_samples, figsize=figsize)
    colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'cyan', 4: 'magenta', 5: 'yellow', 6: 'blacks'}

    for gi, gn in enumerate(sampled_groups):
        g = gb.get_group(gn)

        not_first = gi != 0

        vec_target = pd.Series(g[target_name].values, index=g[date_name])

        def apply_target_patches(ax, ymin, ymax):
            for j in range(len(vec_target)):
                if not np.isnan(vec_target[j]):
                    dt = mdates.date2num(vec_target.index.to_pydatetime()[j])
                    ax.add_patch(patches.Rectangle((dt - 0.5, ymin), 1, ymax - ymin,
                                                   color=colors[vec_target[j]], alpha=.3))

        # Plot the categorical variables first
        for i, cc in enumerate(cat_cols):
            vec = pd.Series(g[cc].values, index=g[date_name])
            cat_mapper = gen_cat_mapper(cat_uniques[cc])
            vec = vec.map(cat_mapper).reset_index()
            if n_samples == 1:
                which_ax = axarr[i]
            else:
                which_ax = axarr[i, gi]
            vec.plot(x=date_name, y=0, style=".", ax=which_ax)
            which_ax.set_ylim([0, len(cat_uniques[cc])])
            which_ax.xaxis.set_visible(False)
            which_ax.legend().set_visible(False)
            if not_first:
                which_ax.yaxis.set_visible(False)
            else:
                tick_order = sorted([cat_mapper[c] for c in cat_mapper])
                which_ax.set_yticks(tick_order)
                labels = [item.get_text() for item in which_ax.get_yticklabels()]
                cat_mapper_inv = {v: k for k, v in cat_mapper.iteritems()}
                for k, tick in enumerate(tick_order):
                    labels[k] = cat_mapper_inv[tick]
                which_ax.set_yticklabels(labels)
            apply_target_patches(which_ax, 0, len(cat_uniques[cc]))

        # Plot the real variables
        for i, rc in enumerate(real_cols):
            ii = i + len(cat_cols)
            vec = pd.Series(g[rc].values, index=g[date_name])
            if n_samples == 1:
                which_ax = axarr[ii]
            else:
                which_ax = axarr[ii, gi]
            vec.plot(ax=which_ax)
            which_ax.set_ylim(feature_ranges[rc])

            if not_first:
                which_ax.yaxis.set_visible(False)

            if i != len(real_cols) - 1:
                which_ax.xaxis.set_visible(False)

            apply_target_patches(which_ax,
                                 feature_ranges[rc][0],
                                 feature_ranges[rc][1])
