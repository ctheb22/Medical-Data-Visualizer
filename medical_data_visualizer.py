import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight']/(df['height']/100)**2) > 25

# 3
df['cholesterol'] = df['cholesterol'] - 1
df.loc[df['cholesterol'] > 0, 'cholesterol'] = 1
df['gluc'] = df['gluc'] - 1
df.loc[df['gluc'] > 0, 'gluc'] = 1

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol','gluc','smoke','alco','active','overweight'])

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable','value']).agg(total=('value','count'))

    # 8
    fig = sns.catplot(data=df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar').fig

    # 9
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11
    df_heat = df.loc[(df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(.975)) &
                     (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(.975)) &
                     (df['ap_lo'] <= df['ap_hi'])]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(df_heat.corr(), dtype=bool))

    # 14
    fig, ax = plt.subplots(1, 1)

    # 15
    heatmap = sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt='.1f')
    fig.set_size_inches(15, 10)

    # 16
    fig.savefig('heatmap.png')
    return fig