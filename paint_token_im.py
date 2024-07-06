import matplotlib
import matplotlib.pyplot as plt
import numpy as np



'''
labels = ['0-12.50%', '12.50%-100%']
men_means = [0.7870239774, 1-0.7870239774]
women_means = [0.9265470005, 1-0.9265470005]
'''



labels = ['Transformer', 'R2Gen', 'CMN']
men_means = [15.44, 15.60, 15.72]
women_means = [9.52, 10.97, 10.73]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(1,3,figsize=(16, 4))
rects1 = ax[1].bar(x - width/2, men_means, width-0.05, label='Normal',color='#8ECFC9')
rects2 = ax[1].bar(x + width/2, women_means, width-0.05, label='Abnormal',color='#FFBE7A')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('BLEU-4',fontsize=15)
#ax.set_xlabel('Model')
#ax.set_title('Scores by group and gender')
ax[1].set_xticks(x)
plt.yticks(fontsize=12)
ax[1].set_xticklabels(labels,fontsize=15)
ax[1].legend(fontsize=15,loc='lower center',bbox_to_anchor=(0.5, -0.30),fancybox=False,ncol=2)




'''
labels = ['0-12.50%', '12.50%-100%']
men_means = [0.7870239774, 1-0.7870239774]
women_means = [0.9265470005, 1-0.9265470005]
'''

labels = ['0-12.50%', '12.50%-100%']
men_means = [0.87374827,0.12625173]
women_means = [0.76930629,0.23069371]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


rects1 = ax[0].bar(x - width/2, men_means, width-0.05, label='Normal',color='#8ECFC9')
rects2 = ax[0].bar(x + width/2, women_means, width-0.05, label='Abnormal',color='#FFBE7A')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Density',fontsize=15)
#ax.set_xlabel('Model')
#ax.set_title('Scores by group and gender')
ax[0].set_xticks(x)
plt.yticks(fontsize=12)
ax[0].set_xticklabels(labels,fontsize=15)
ax[0].legend(fontsize=15,loc='lower center',bbox_to_anchor=(0.5, -0.30),fancybox=False,ncol=2)


labels = ['Transformer', 'R2Gen', 'CMN']
men_means = [2.14, 2.52, 2.23]
women_means = [48.92, 52.01, 45.60]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

rects1 = ax[2].bar(x - width/2, men_means, width-0.05, label='low freq',color='#8ECFC9')
rects2 = ax[2].bar(x + width/2, women_means, width-0.05, label='high freq',color='#FFBE7A')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[2].set_ylabel('F1',fontsize=15)
#ax.set_xlabel('Model')
#ax.set_title('Scores by group and gender')
ax[2].set_xticks(x)
plt.yticks(fontsize=12)
ax[2].set_xticklabels(labels,fontsize=15)
ax[2].legend(fontsize=15,loc='lower center',bbox_to_anchor=(0.5, -0.30),fancybox=False,ncol=2)

ax[0].set_title('(1)',fontsize=15)
ax[1].set_title('(2)',fontsize=15)
ax[2].set_title('(3)',fontsize=15)


plt.savefig('label_im_performance.pdf',bbox_inches='tight',pad_inches=5)
plt.show()