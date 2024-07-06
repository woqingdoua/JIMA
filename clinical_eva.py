import pandas as pd
import torch
from pretrained_classification_model import ClassificationModel
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import json

def bert_encode(x):

    xs = []
    xs_mask = []
    for i in range(len(x)):
        pre_reports = bert_tokenizer.encode(x[i])
        ms = [1] * len(pre_reports)
        xs.append(pre_reports)
        xs_mask.append(ms)

    max_length = max([len(i) for i in xs])

    bert_targets = np.zeros((len(xs), max_length), dtype=int)
    bert_targets_masks = np.zeros((len(xs), max_length), dtype=int)

    for i, report_ids in enumerate(xs):
        bert_targets[i, :len(report_ids)] = report_ids

    for i, report_masks in enumerate(xs_mask):
        bert_targets_masks[i, :len(report_masks)] = report_masks

    bert_targets, bert_targets_masks = torch.tensor(bert_targets), torch.tensor(bert_targets_masks)

    return bert_targets, bert_targets_masks


def clinical_acc(prediction, label):

    pre_index, pre_mask = bert_encode(prediction)
    #index, mask = bert_encode(truth)
    pre_label = (classificationmodel(pre_index,mask=pre_mask).squeeze(-1)>0.5).float()
    #label = (classificationmodel(index,mask=mask).squeeze(-1)>0.5).float()
    label = torch.tensor(label)
    right = torch.sum(pre_label*label)
    precision = right/torch.sum(pre_label)
    recall = right/torch.sum(label)
    F = 2*precision*recall/(precision+recall)
    print(f'precision:{precision},recall:{recall},F:{F}')

'''
data = pd.read_csv('/home/ywu10/vilmedic/vilmedic_mimic.csv')
truth = json.loads(open('/home/ywu10/Documents/R2Gen/data/mimic_cxr/annotation_label.json', 'r').read())
truth = [i['label'] for i in truth['test']]
prediction = data['prediction']
'''
with open("/home/ywu10/Documents/multimodel/results/testrun_gt_results_2022-10-17-20-41.txt", "r") as f:
    truth_ = f.readlines()

with open("/home/ywu10/Documents/multimodel/results/testrun_pre_results_2022-10-17-20-41.txt", "r") as f:
    prediction_ = f.readlines()

truth = []
for i in truth_:
    if "Impression:" in i:
        truth.append(i[12:])

prediction = []
for i in prediction_:
    if "Impression:" in i:
        prediction.append(i[12:])

data = pd.DataFrame({'Report Impression':truth})
data.to_csv('/home/ywu10/Documents/multimodel/results/truth.csv')

data = pd.DataFrame({'Report Impression':prediction})
data.to_csv('/home/ywu10/Documents/multimodel/results/prediction.csv')

'''
classificationmodel = ClassificationModel()
#pretrained_params = torch.load('/home/ywu10/Documents/r2genbaseline/classification/best_classifier_mimic.pkl')
bert_tokenizer = BertTokenizer.from_pretrained("/home/ywu10/Documents/r2genbaseline/classification")

clinical_acc(prediction,truth)
'''

