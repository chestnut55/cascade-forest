import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

databases = ['cirrhosis', 't2d', 'obesity']

for database in databases:
    fig, ax1 = plt.subplots()
    width = .4

    imp = pd.Series.from_csv(database + '_importance.csv').values * 100
    healthy_abundance = pd.Series.from_csv(database + '_healthy.csv').values
    disease_abundance = pd.Series.from_csv(database + '_disease.csv').values

    names = pd.Series.from_csv(database + '_healthy.csv').index.tolist()


    name_list = []
    for name in names:
        name = name[name.rfind('|')+1:len(name)]
        name_list.append(name)

    for s in range(len(name_list)):
        if 's__' in names[s]:
            names[s] = ' '.join(names[s].split('s__')[-1].split('|t__')[0].split('_'))
        elif 'g__' in names[s]:
            names[s] = 'g: ' + ' '.join(names[s].split('g__')[-1].split('_'))
        elif 'f__' in names[s]:
            names[s] = 'f: ' + ' '.join(names[s].split('f__')[-1].split('_'))
        if 'unclassified' in names[s]:
            names[s] = names[s][:-12] + 'spp.'

    green = sns.xkcd_rgb["medium green"]
    red = sns.xkcd_rgb["pale red"]
    blue = sns.xkcd_rgb["windows blue"]

    ind = np.arange(len(names)) * 1.5

    ax1.barh(ind + width, healthy_abundance, width, color=green)
    ax1.barh(ind + 2 * width, disease_abundance, width, color=red)
    ax1.set_xlabel('Healthy(in green) and diseased(in red) average relative abundance[%]', color='black')

    ax2 = ax1.twiny()
    ax2.barh(ind, list(imp)[:10], width, color=blue)
    ax2.set_yticks(ind + width)
    ax2.set_yticklabels(names)
    ax2.set_xlabel('Relative importance(in blue)[%]', color='black')
    ax2.tick_params('x', colors='black')

    ax2.tick_params('y', colors='black')
    ax2.invert_yaxis()
    ax1.set_xscale('log')

    fig.subplots_adjust(wspace=1.5)
    fig.suptitle(database + ' dataset',y=1, size=14)

    fig.savefig(database+'.png', bbox_inches='tight')
    # plt.show()
