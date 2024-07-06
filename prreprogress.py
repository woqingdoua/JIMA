import pandas as pd
import json
import re

file = '/home/ywu10/Documents/R2Gen/data/indiana_reports.csv'
image_path = '/home/ywu10/Documents/R2Gen/data/iu_xray/annotation.json'
data_label = pd.read_csv(file)
labels = []
id = []
findings = []
count = 0

data = json.loads(open(image_path, 'r').read())
for tt in data:
    index = 0
    for dd in data[tt]:
        uid = int(dd['id'][3:re.search('_',dd['id']).span()[0]])
        label = list(data_label[data_label['uid']==uid]['MeSH'])[0]
        if label == 'normal':
            data[tt][index]['label'] = 0
        else:
            data[tt][index]['label'] = 1
        index += 1

with open("/home/ywu10/Documents/R2Gen/data/iu_xray/annotation_label.json","w") as f:
    json.dump(data,f)
