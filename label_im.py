import json

import matplotlib.pyplot as plt
import numpy as np

ann_path = '/home/ywu10/Documents/R2Gen/data/mimic_cxr/annotation_label.json'
data = json.loads(open(ann_path, 'r').read())
label = [i['label'] for i in data['train']]
label = np.array(label)
#ad = sum([i['label'] for i in data['train']])/len([i['label'] for i in data['train']])
#nor = 1-ad
dis = np.sum(label,0)/np.sum(label)
#x = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Lesion","Lung Opacity","Edema","Consolidation","Pneumonia",\
#     "Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices"]
x = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
plt.bar(x,dis,color='#8ECFC9')
plt.ylabel('Percentage')
plt.savefig('label_im.pdf')
