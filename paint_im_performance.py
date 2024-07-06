import matplotlib
import matplotlib.pyplot as plt
import numpy as np




labels = ['0-12.50%', '12.50%-100%']
men_means = [0.7870239774, 1-0.7870239774]
women_means = [0.9265470005, 1-0.9265470005]
'''

labels = ['Normal', 'Abnormal']
men_means = [0.3296, 0.6704]
women_means = [0.1397, 0.8603]
'''

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width-0.05, label='IU X-ray',color='#8ECFC9')
rects2 = ax.bar(x + width/2, women_means, width-0.05, label='MIMIC-CXR',color='#FFBE7A')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Density',fontsize=15)
ax.set_xlabel('Top',fontsize=15)
#ax.set_title('Scores by group and gender')
ax.set_xticks(x)
plt.yticks(fontsize=12)
ax.set_xticklabels(labels,fontsize=15)
ax.legend(fontsize=15)

#plt.show()
plt.savefig('token_im.pdf')