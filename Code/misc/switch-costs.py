#!/usr/bin/env python
# coding: utf-8

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = {}

path = '../experiments/{paper}/e{exp_num}/Output/{mode}/13b9435ca5add3409d7fb2cbc6f836a0/Data/{fname}.csv'

for i in range(1, 4):
    data[f'Hemed & Eitam, 2022 - E{i}'] = {}
    data[f'Hemed & Eitam, 2022 - E{i}']['raw'] = pd.read_csv(
        path.format(paper='relevance', exp_num=i, fname='raw', mode='amended')
    ).assign(Paper='Hemed & Eitam, 2022', Experiment=f'{i}')
    data[f'Hemed & Eitam, 2022 - E{i}']['agg'] = [
        pd.read_csv(path.format(paper='relevance', exp_num=i, fname='post_aggregation', mode=j)
                    ).assign(Paper='Hemed & Eitam, 2022', Experiment=f'{i}')
        for j in ['confounded', 'amended']
    ]

data['Hemed & Eitam, 2022 - E1']['raw'] = data['Hemed & Eitam, 2022 - E1']['raw'].query(
    'feedback == "CONTROL FEEDBACK"')
data['Hemed & Eitam, 2022 - E1']['agg'][0] = data['Hemed & Eitam, 2022 - E1']['agg'][0].query(
    'feedback == "CONTROL FEEDBACK"')
data['Hemed & Eitam, 2022 - E1']['agg'][1] = data['Hemed & Eitam, 2022 - E1']['agg'][1].query(
    'feedback == "CONTROL FEEDBACK"')

for i in range(3, 6):
    data[f'Current - E{i}'] = {}
    data[f'Current - E{i}']['raw'] = pd.read_csv(
        path.format(paper='revisited', exp_num=i, mode='amended', fname='raw')
    ).assign(Paper='Current', Experiment=f'{i}')

    data[f'Current - E{i}']['agg'] = [
        pd.read_csv(path.format(paper='revisited', exp_num=i, fname='post_aggregation', mode=j)
                    ).assign(Paper='Current', Experiment=f'{i}')
        for j in ['confounded', 'amended']
    ]

df = pd.concat([j['raw'].assign(src=k) for k, j in data.items()])

df_amended = pd.concat([j['agg'][1].assign(src=k) for k, j in data.items()])
df_confounded = pd.concat([j['agg'][0].assign(src=k) for k, j in data.items()])

palette = sns.color_palette(['purple', 'green', ])
plt.rcParams["font.family"] = "DejaVu Sans"

# Create a mapping between categories and color indices
category_colors = {5: 0, 10: 1}


# Create a custom color palette function
def color_map(category):
    return palette[category_colors[category]]


df = df.sort_values('src')

plot_data = df.loc[
    (df['output_correct'] == 1) &
    (
            ((df['trials_since_last_probe'] == 0) & (df['probed_trial'])) |
            ((df['trials_since_last_probe'].between(1, 5)) & ~df['probed_trial'])
    )
    ].groupby(
    ['src', 'patch_length', 'unique_participant', 'trials_since_last_probe']
)['RT'].agg(['mean', 'count']).reset_index()

plot_data['patch_length'] = plot_data['patch_length'].astype(int)
data_order = plot_data['src'].unique().reshape((2, 3)).T.reshape(-1)

plot_data['src_categorical'] = pd.Categorical(plot_data['src'], categories=data_order, ordered=True)
plot_data = plot_data.sort_values('src_categorical')

plot_data[['trials_since_last_probe', 'patch_length']] = plot_data[
    ['trials_since_last_probe', 'patch_length']
].astype(int)

main_text_plot_data = plot_data.loc[plot_data['src'].isin(['Current - E3', 'Current - E4'])]
main_text_amended = df_amended.loc[df_amended['src'].isin(['Current - E3', 'Current - E4'])]
main_text_confounded = df_confounded.loc[df_confounded['src'].isin(['Current - E3', 'Current - E4'])]

fig, axs = plt.subplots(2, 2, figsize=(8.5, 5.5), sharex='col', sharey='col',
                        gridspec_kw={'width_ratios': [2, 1]})

# Draw switch cost
for (expname, expdf), ax in zip(main_text_plot_data.groupby('src'), axs[:, 1]):
    g = sns.pointplot(
        ax=ax,
        data=expdf.query('trials_since_last_probe > 0'),
        x="trials_since_last_probe",
        y="mean",
        hue="patch_length",
        # kind="point",
        dodge=0.2,
        palette=palette,
        join=False,
        # legend=False,
        errwidth=0.4,
        capsize=0.2,
        markers=''
    )
    ax.set(xlabel='', ylabel='')
    ax.legend().remove()

custom_lines = [
    plt.Line2D([0], [0], marker='_', color='purple', linestyle=''),
    plt.Line2D([0], [0], marker='_', color='green', linestyle='')
]

axs[0, 1].legend(
    custom_lines,
    ['5-Trial\nCycle', '10-Trial\nCycle'],
    handlelength=1,
    ncol=2,
    facecolor='gainsboro'
)

axs[1, 1].set_xlabel('Task trials since\nlast attentional probe', fontsize=14)

for (expname, expdf), ax in zip(main_text_confounded.groupby('src'), axs[:, 0]):
    _exp = expdf.copy()
    _exp.loc[_exp['prior'] == False, 'context'] = 8 - _exp.loc[_exp['prior'] == False, 'context']

    sns.pointplot(
        ax=ax,
        data=_exp,
        join=False,
        dodge=0.2,
        x='context',
        palette=palette,
        hue='patch_length',
        y='RT',
        # legend=False,
        errwidth=0.4,
        capsize=0.2,
        markers='x',
        errorbar=('ci', False)
    )
    ax.legend().remove()

    means = _exp.groupby(['patch_length', 'context'])['RT'].mean().values.reshape(2, 8)

    for i, c, d in zip([0, 1], ['purple', 'green'], [-0.1, 0.1]):
        ax.plot(np.arange(0, 8) + d, means[i], c=c, alpha=0.4, lw=0.2)
        ax.plot(np.arange(3, 5) + d, means[i, 3:5], c=c, alpha=0.75, lw=0.75)

for (expname, expdf), ax in zip(main_text_amended.groupby('src'), axs[:, 0]):
    _exp = expdf.copy()
    _exp.loc[_exp['prior'] == False, 'context'] = 8 - _exp.loc[_exp['prior'] == False, 'context']

    means = _exp.groupby(['patch_length', 'context'])['RT'].mean().values.reshape(2, 8)

    for i, c, d in zip([0, 1], ['purple', 'green'], [-0.1, 0.1]):
        ax.plot(np.arange(0, 8) + d, means[i], c=c, alpha=0.4, lw=0.2)
        ax.plot(np.arange(3, 5) + d, means[i, 3:5], c=c, alpha=0.75, lw=0.75)

    sns.pointplot(
        ax=ax,
        data=_exp,
        join=False,
        dodge=0.2,
        x='context',
        palette=palette,
        hue='patch_length',
        y='RT',
        # legend=False,
        errwidth=0.4,
        capsize=0.2,
        markers='.',
    )
    ax.legend().remove()


    ax.set(xlabel='', ylabel='', xticklabels=[])
    ax.legend().remove()

for ax in axs[:, 0]:
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=(2.75, ax.get_ylim()[0]),
        width=1.5,
        height=ax.get_ylim()[1] - ax.get_ylim()[0],
        color='yellow',
        alpha=0.75,
        zorder=0
    ))

axs[0, 0].set_title("Hemed et al., 2020 - E1 \n E3 in current paper (see supplementary materials)")
axs[1, 0].set_title("Hemed et al., 2020 - E2 (E4, here)")

axs[0, 0].annotate("Contingent action-effect\non trial N-1", [1.25, 455], ha='center', weight='bold', fontsize=9)
axs[0, 0].annotate("No contingent action-effect\non trial N-1", [5.75, 455], ha='center', weight='bold', fontsize=9)

custom_lines = custom_lines + [
    plt.Line2D([0], [0], marker='.', color='black', linestyle=''),
    plt.Line2D([0], [0], marker='x', color='black', linestyle='')
]

axs[1, 0].legend(
    custom_lines,
    ['5-Trial\nCycle', '10-Trial\nCycle', 'Corrected\n(Current)', 'Uncorrected\n(Hemed et al., 2020)'],
    handlelength=0,
    ncol=4,
    facecolor='gainsboro',
    loc='lower right',
    fontsize=10
)

axs[1, 0].set_xlabel('Sum of contingent action-effect\noccurrences in trials N-4 through N-2', fontsize=14)
axs[1, 0].set_ylabel('RT (ms), 95%-CI', fontsize=14)

axs[1, 0].set_xticklabels([0, 1, 2, 3, 3, 2, 1, 0])

fig.tight_layout()
fig.savefig('switch-cost-fig.png', dpi=500)
