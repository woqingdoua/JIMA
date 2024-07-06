import pickle
import os
import json
import csv
import numpy as np
import pandas as pd

'''
ann = json.loads(open('/home/ywu10/Documents/R2Gen/data/mimic_cxr/annotation.json', 'r').read())
start = np.arange(0,len(ann['train']),5000,dtype=int)
report = [[i['report'].replace('\n', '')] for i in ann['train']]
for i in range(len(start)):
    file_name = "/home/ywu10/Documents/r2genbaseline/data/mimic_cxr_report_" + str(i) + ".csv"
    with open(file_name,"w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(report[start[i]:start[i]+5000])
'''

'''
ann = json.loads(open('/home/ywu10/Documents/R2Gen/data/mimic_cxr/annotation.json', 'r').read())
file_name = "/home/ywu10/Documents/r2genbaseline/data/mimic_cxr_report_" + str(55) + ".csv"
report = [[i['report'].replace('\n', '')] for i in ann['train']]
with open(file_name,"w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(report[-790:])

report = [[i['report'].replace('\n', '')] for i in ann['val']]
#for i in range(len(start)):
file_name = "/home/ywu10/Documents/r2genbaseline/data/mimic_cxr_report_" + str(56) + ".csv"
with open(file_name,"w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(report)

report = [[i['report'].replace('\n', '')] for i in ann['test']]
#for i in range(len(start)):
file_name = "/home/ywu10/Documents/r2genbaseline/data/mimic_cxr_report_" + str(57) + ".csv"
with open(file_name,"w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(report)
'''

    

save_file = '/home/ywu10/Documents/R2Gen/data/mimic_cxr/annotation_label.json'
data = json.loads(open('/home/ywu10/Documents/R2Gen/data/mimic_cxr/annotation.json', 'r').read())
n = np.arange(0,55,dtype=int)
root = '/home/ywu10/Documents/r2genbaseline/data/'

index = 0
total_label = 0
unlabel_report = []
for i in n:
    label_f = root + 'mimic_cxr_label_' + str(i) + '.csv'
    label = pd.read_csv(label_f,usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]).values
    report = pd.read_csv(label_f)['Reports'].values
    label[np.isnan(label)] = 0
    label = label > 0
    label = label.tolist()
    total_label += len(label)
    for ll in range(len(label)):
        try:
            if report[ll] == data['train'][index]['report'].replace('\n',''):
                data['train'][index]['label'] = label[ll]
            else:
                unlabel_report.append(data['train'][index]['report'].replace('\n',''))
            index += 1
        except:
            print('a')


label_f = root + 'mimic_cxr_label_' + str(56) + '.csv'
label = pd.read_csv(label_f,usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]).values
report = pd.read_csv(label_f)['Reports'].values
label[np.isnan(label)] = 0
label = label > 0
label = label.tolist()
total_label += len(label)
for ll in range(len(label)):
    if report[ll] == data['val'][ll]['report'].replace('\n',''):
        data['val'][ll]['label'] = label[ll]
    else:
        unlabel_report.append(data['val'][ll]['report'].replace('\n',''))


label_f = root + 'mimic_cxr_label_' + str(57) + '.csv'
label = pd.read_csv(label_f,usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]).values
report = pd.read_csv(label_f)['Reports'].values
label[np.isnan(label)] = 0
label = label > 0
label = label.tolist()
total_label += len(label)
for ll in range(len(label)):
    if report[ll] == data['test'][ll]['report'].replace('\n',''):
        data['test'][ll]['label'] = label[ll]
    else:
        unlabel_report.append(data['test'][ll]['report'].replace('\n',''))


with open(save_file,"w") as f:
    json.dump(data,f)




