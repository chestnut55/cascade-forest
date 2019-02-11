import matplotlib.pyplot as plt
import pandas as pd

cirrhosis = pd.read_csv('cirrhosis', index_col=0)
t2d = pd.read_csv('t2d', index_col=0)
obesity = pd.read_csv('obesity', index_col=0)
fig = plt.figure(figsize=(10, 2))
axes = fig.subplots(1, 3, sharey=True, sharex=True)
cirrhosis.plot(ax=axes[0], linestyle='--', marker='*', color=['r', 'b', 'g', 'y'])
axes[0].set_title('Cirrhosis')

axes[0].set_ylabel('Stability')
axes[0].set_xlabel('Features left(%)')
# axes[0].set_yticks([0.2, 0.4, 0.6, 0.8])
# axes[0].set_xticks(range(9),['2','3','4','5','10','20','30','40','50'])
# axes[0].set_xticklabels(['2%','3%','4%','5%','10%','20%','30%','40%','50%'])



t2d.plot(ax=axes[1], linestyle='--', marker='*', color=['r', 'b', 'g', 'y'], legend=False)
axes[1].set_title('T2d')
axes[1].set_xlabel('Features left(%)')

obesity.plot(ax=axes[2], linestyle='--', marker='*', color=['r', 'b', 'g', 'y'], legend=False)
axes[2].set_title('Obesity')
axes[2].set_xlabel('Features left(%)')

# plt.subplots_adjust(wspace=0,hspace=0)
# for tick in axes[0].get_xticklabels():
#     tick.set_visible(True)

fig.tight_layout()
plt.show()
