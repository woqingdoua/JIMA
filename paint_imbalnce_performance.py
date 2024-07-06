import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['R2Gen', 'CMN', 'WCL']
men_means = [20.93, 18.72, 18.71]
women_means = [15.04, 14.36, 13.81]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(1,4,figsize=(20, 4))
rects1 = ax[0].bar(x - width/2, men_means, width-0.05, label='Normal',color='#8ECFC9')
rects2 = ax[0].bar(x + width/2, women_means, width-0.05, label='Abnormal',color='#FFBE7A')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('BLEU-4',fontsize=15)
#ax.set_xlabel('Model')
#ax.set_title('Scores by group and gender')
ax[0].set_xticks(x)
plt.yticks(fontsize=12)
ax[0].set_xticklabels(labels,fontsize=15)
ax[0].legend(fontsize=15,loc='lower center',bbox_to_anchor=(0.5, -0.30),fancybox=False,ncol=2)


labels = ['R2Gen', 'CMN', 'WCL']
men_means = [4.46, 5.88,5.29]
women_means = [62.73 , 55.86, 60.23]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

rects1 = ax[1].bar(x - width/2, men_means, width-0.05, label='low freq',color='#8ECFC9')
rects2 = ax[1].bar(x + width/2, women_means, width-0.05, label='high freq',color='#FFBE7A')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('F1',fontsize=15)
#ax.set_xlabel('Model')
#ax.set_title('Scores by group and gender')
ax[1].set_xticks(x)
plt.yticks(fontsize=12)
ax[1].set_xticklabels(labels,fontsize=15)
ax[1].legend(fontsize=15,loc='lower center',bbox_to_anchor=(0.5, -0.30),fancybox=False,ncol=2)



# ----------
labels = ['R2Gen', 'CMN', 'WCL']
men_means = [15.60, 15.72, 13.71]
women_means = [10.97, 8.73, 10.26]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


rects1 = ax[2].bar(x - width/2, men_means, width-0.05, label='Normal',color='#8ECFC9')
rects2 = ax[2].bar(x + width/2, women_means, width-0.05, label='Abnormal',color='#FFBE7A')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[2].set_ylabel('BLEU-4',fontsize=15)
#ax.set_xlabel('Model')
#ax.set_title('Scores by group and gender')
ax[2].set_xticks(x)
plt.yticks(fontsize=12)
ax[2].set_xticklabels(labels,fontsize=15)
ax[2].legend(fontsize=15,loc='lower center',bbox_to_anchor=(0.5, -0.30),fancybox=False,ncol=2)


labels = ['R2Gen', 'CMN', 'WCL']
men_means = [2.52, 2.23, 2.91]
women_means = [52.01, 45.60,48.60]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

rects1 = ax[3].bar(x - width/2, men_means, width-0.05, label='low freq',color='#8ECFC9')
rects2 = ax[3].bar(x + width/2, women_means, width-0.05, label='high freq',color='#FFBE7A')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[3].set_ylabel('F1',fontsize=15)
#ax.set_xlabel('Model')
#ax.set_title('Scores by group and gender')
ax[3].set_xticks(x)
plt.yticks(fontsize=12)
ax[3].set_xticklabels(labels,fontsize=15)
ax[3].legend(fontsize=15,loc='lower center',bbox_to_anchor=(0.5, -0.30),fancybox=False,ncol=2)

ax[0].set_title('(1) IU X-ray',fontsize=15)
ax[1].set_title('(2) IU X-ray',fontsize=15)
ax[2].set_title('(3) MIMIC-CXR',fontsize=15)
ax[3].set_title('(4) MIMIC-CXR',fontsize=15)


plt.savefig('label_im_performance.pdf',bbox_inches='tight',pad_inches=5)
plt.show()