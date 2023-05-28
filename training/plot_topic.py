import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

#sns.set_style("whitegrid")
sns.set_style("whitegrid", {'font.family':'serif'})
sem_acc_trf = [0.068, 0.084, 0.239, 0.303, 0.377]
mauve_trf = [0.95, 0.94, 0.81, 0.62, 0.41]

SMALL_SIZE = 13
MEDIUM_SIZE = 14
BIGGER_SIZE = 15

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



sem_acc_backpack = [0.072, 0.121, 0.243, 0.353]
mauve_backpack = [0.92, 0.91, 0.90, 0.83]

#sns.set_style("whitegrid")
#sns.boxplot(data=data, palette="deep")
sns.despine(left=True)

plt.xlabel('17-Topic Average Control Success')
plt.ylabel('Overall MAUVE with OpenWebText')
#plt.plot(mauve_trf, sem_acc_trf)
#plt.plot(mauve_backpack, sem_acc_backpack)


#plt.annotate("Unmodified Transformer", (0.068, 0.95))
sns.despine()

#plt.annotate("Unmodified Backpack", (0.074, 0.93))

plt.annotate("Unmodified Transformer",
            xy=(0.065, 0.95), xycoords='data',
            xytext=(0.06, 0.65), textcoords='data',
            arrowprops=dict(arrowstyle="->", color='black', linestyle='--',
                            connectionstyle="arc3"),
             c='#593C8F',
            )
plt.annotate("Unmodified Backpack",
            xy=(0.072, 0.925), xycoords='data',
            xytext=(0.12, 0.75), textcoords='data',
            arrowprops=dict(arrowstyle="->", color='black', linestyle='--',
                            connectionstyle="arc3"),
            c='#DB5461',
            )



            

plt.title('Topic Control in Generation', fontsize=BIGGER_SIZE)

plt.plot(sem_acc_trf, mauve_trf, label='Transformer+PPLM', marker='s',linewidth =2, c='#593C8F')
plt.plot(sem_acc_backpack, mauve_backpack, label='Backpack+sense control', marker='o', linewidth=2, c='#DB5461')
plt.legend()
plt.tight_layout()
plt.savefig('plt.pdf')
