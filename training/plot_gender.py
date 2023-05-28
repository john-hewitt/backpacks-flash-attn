import numpy as np
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
from collections import OrderedDict

nrows, ncols = 1, 3
fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows), sharey=True)
sns.set() # Setting seaborn as default style even if use only matplotlib
#matplotlib.rcParams['font.family'] = 'serif'




sns.set_style("whitegrid", {'font.family':'serif'})
#sem_acc_trf = [0.068, 0.084, 0.239, 0.303, 0.377]
#mauve_trf = [0.95, 0.94, 0.81, 0.62, 0.41]

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#font = {'family' : 'normal',
#        'size'   : 12}
#
#matplotlib.rc('font', **font)



#sem_acc_backpack = [0.074, 0.121, 0.243, 0.353]
#mauve_backpack = [0.93, 0.91, 0.90, 0.83]

#sns.set_style("whitegrid")
#sns.boxplot(data=data, palette="deep")
sns.despine(left=True)

#plt.xlabel('21-Topic Average Control Success')
#plt.ylabel('Overall MAUVE with OpenWebText')
#plt.plot(mauve_trf, sem_acc_trf)
#plt.plot(mauve_backpack, sem_acc_backpack)


#plt.annotate("Unmodified Transformer", (0.068, 0.95))
sns.despine()


p1 = OrderedDict({' she': 0.24740535020828247, ' the': 0.15123586356639862, ' he': 0.1150541827082634, ' they': 0.04540923982858658, ' I': 0.03648734837770462, ' a': 0.026904083788394928, ' there': 0.026904083788394928, ' her': 0.02337467297911644, ' it': 0.012708589434623718,})# ' we': 0.012032250873744488})

p0 = OrderedDict({' he': 0.3735756278038025, ' the': 0.11755078285932541, ' she': 0.0659400150179863, ' his': 0.039683401584625244, ' they': 0.03529514744877815, ' I': 0.03114785999059677, ' a': 0.02444823458790779, ' there': 0.021075695753097534, ' it': 0.010933926329016685,})# ' we': 0.010933926329016685})

p07 = OrderedDict({' she': 0.1779058426618576, ' he': 0.17514768242835999, ' the': 0.1498117446899414, ' they': 0.04498164355754852, ' I': 0.037291090935468674, ' a': 0.02792973630130291, ' there': 0.026859765872359276, ' her': 0.017753221094608307, ' his': 0.016808411106467247,})# ' it': 0.01298853475600481})

#p1 = OrderedDict({' she': 0.24740535020828247, ' the': 0.15123586356639862, ' he': 0.1150541827082634, ' they': 0.04540923982858658, ' I': 0.03648734837770462, ' a': 0.026904083788394928, ' there': 0.026904083788394928, ' her': 0.02337467297911644, ' it': 0.012708589434623718, ' we': 0.012032250873744488})
#
#p0 = OrderedDict({' he': 0.3735756278038025, ' the': 0.11755078285932541, ' she': 0.0659400150179863, ' his': 0.039683401584625244, ' they': 0.03529514744877815, ' I': 0.03114785999059677, ' a': 0.02444823458790779, ' there': 0.021075695753097534, ' it': 0.010933926329016685, ' we': 0.010933926329016685})
#
#p07 = OrderedDict({' she': 0.1779058426618576, ' he': 0.17514768242835999, ' the': 0.1498117446899414, ' they': 0.04498164355754852, ' I': 0.037291090935468674, ' a': 0.02792973630130291, ' there': 0.026859765872359276, ' her': 0.017753221094608307, ' his': 0.016808411106467247, ' it': 0.01298853475600481})


axs[0].bar(x=list(p0.keys()), height=list(map(lambda x:p0[x], p0)), color='#DB5461')
axs[0].set_ylabel('Probability')
axs[1].bar(x=list(p07.keys()), height=list(map(lambda x:p07[x], p07)), color='#593C8F')
axs[2].bar(x=list(p1.keys()), height=list(map(lambda x:p1[x], p1)), color='#171738')

#sns.barplot(x=list(p1.keys()), y=list(map(lambda x:p1[x], p1)))
#sns.barplot(x=list(p0.keys()), y=list(map(lambda x:p0[x], p1)))
#sns.barplot(x=list(p07.keys()), y=list(map(lambda x:p07[x], p1)))

axs[0].tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
axs[1].tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
axs[2].tick_params(axis='both', which='major', labelsize=SMALL_SIZE)

axs[0].set_ylabel('Probability', fontsize=MEDIUM_SIZE)

axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45)
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45)
axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=45)


#plt.suptitle('Topic Control in Generation', fontsize=BIGGER_SIZE)
#plt.legend()
plt.tight_layout()
plt.savefig('plt.png', dpi=500)

print(p1, p0, p07)
exit()


#plt.annotate("Unmodified Backpack", (0.074, 0.93))

plt.annotate("Unmodified Transformer",
            xy=(0.068, 0.95), xycoords='data',
            xytext=(0.06, 0.65), textcoords='data',
            arrowprops=dict(arrowstyle="->", color='black', linestyle='--',
                            connectionstyle="arc3"),
             c='#593C8F',
            )
plt.annotate("Unmodified Backpack",
            xy=(0.074, 0.93), xycoords='data',
            xytext=(0.11, 0.75), textcoords='data',
            arrowprops=dict(arrowstyle="->", color='black', linestyle='--',
                            connectionstyle="arc3"),
            c='#DB5461',
            )



            


plt.plot(sem_acc_trf, mauve_trf, label='Transformer+PPLM', marker='s',linewidth =2, c='#593C8F')
plt.plot(sem_acc_backpack, mauve_backpack, label='Backpack+SenseControl', marker='o', linewidth=2, c='#DB5461')
plt.tight_layout()
plt.savefig('plt.pdf')
