import json
import pandas as pd

ann_path = '/home/ywu10/Documents/R2Gen/data/mimic_cxr/annotation_label.json' #/home/ywu10/Documents/R2Gen/data/iu_xray/annotation_label.json'
ann = json.loads(open(ann_path, 'r').read())
train = ann['train']
val = ann['val']
test = ann['test']

image_path = []
label = []
report = []

ll = ['Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices','No Finding']

for i in ann:
    for ii in ann[i]:
        image_path.append(ii['image_path'])
        report.append(ii['report'])
        a = []
        for l in range(len(ii['label'])):
            if ii['label'][l] == True:
                a.append(ll[l])
        label.append(a)

data = {'image_path':image_path,'report':report,'label':label}
data = pd.DataFrame(data)
data.to_csv('MIMIC_label.csv')