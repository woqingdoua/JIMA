import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font1 = {'family' : 'Times New Roman',
         'size':15,}

labels = ['0-12.50%', '12.50%-100%']


men_means = [0.9195313, 0.0804687]
women_means = [0.80648833, 0.19351167]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(1,2,figsize=(10, 4))



x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


rects1 = ax[0].bar(10 - width/2, men_means, width-0.05, label='Normal',color='#8ECFC9')
rects2 = ax[0].bar(x + width/2, women_means, width-0.05, label='Abnormal',color='#FFBE7A')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Density',fontsize=15)
#ax.set_xlabel('Model')
#ax.set_title('Scores by group and gender')
ax[0].set_xticks(x)
plt.yticks(fontsize=12)
ax[0].set_xticklabels(labels,fontsize=15)
ax[0].legend(loc='lower center',bbox_to_anchor=(0.5, -0.30),fancybox=False,ncol=2, prop=font1)


men_means = [0.87374827,0.12625173]
women_means = [0.76930629,0.23069371]

rects1 = ax[1].bar(x - width/2, men_means, width-0.05, label='Normal',color='#8ECFC9')
rects2 = ax[1].bar(x + width/2, women_means, width-0.05, label='Abnormal',color='#FFBE7A')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('Density',fontsize=15)
#ax.set_xlabel('Model')
#ax.set_title('Scores by group and gender')
ax[1].set_xticks(x)
plt.yticks(fontsize=12)
ax[1].set_xticklabels(labels,fontsize=15)
ax[1].legend(loc='lower center',bbox_to_anchor=(0.5, -0.30),fancybox=False,ncol=2,prop=font1)

ax[0].set_title('(1) IU X-ray',fontsize=15)
ax[1].set_title('(2) MIMIC-CXR',fontsize=15)


plt.savefig('data_label_imbalance.pdf',bbox_inches='tight',pad_inches=5)
plt.show()