from modules.metrics import compute_scores
import pandas as pd
import json
import re
from collections import Counter
import numpy as np
from nltk.translate.meteor_score import meteor_score


def imbalanced_eval(token2idx_,pre,tgt,n):

    #words = dict(sorted(dict(self.model.tokenizer.counter).items(), key=lambda x: x[1]))
    words = [w for w in token2idx_][:-2]
    recall_ = []
    precision_ = []
    right_ = []
    gap = len(words)//n
    for index in range(0,len(words)-gap,gap):
        right = 0
        recall = 0
        precision = 0
        for i in range(len(tgt)):
            a = [j for j in tgt[i].split() if j in words[index:index+gap]]
            b = [j for j in pre[i].split() if j in  words[index:index+gap]]
            right += len([j for j in a if j in b])
            recall += len(a)
            precision += len(b)
        recall_.append(recall)
        precision_.append(precision)
        right_.append(right)
    r = np.array(right_)/np.array(recall_)
    print(f'recall:{r}')
    p = np.array(right_)/np.array(precision_)
    print(f'precision:{p}')
    print(precision_)
    print(recall_)
    f1  =  2*r*p/(r+p)
    f1  =  np.nan_to_num(f1)
    print(f'f1:{f1[0]},{sum(f1[1:])}')

def create_vocabulary(ann,clean_report,threshold):
    total_tokens = []

    for example in ann['train']:
        tokens = clean_report(example['report']).split()
        for token in tokens:
            total_tokens.append(token)

    counter = Counter(total_tokens)
    vocabs = dict(sorted(counter.items(),key = lambda x:x[1],reverse=True))
    vocab =  [i  for  i in vocabs if  vocabs[i]>= threshold]
    #vocab.sort()
    token2idx, idx2token = {}, {}
    for idx, token in enumerate(vocab):
        token2idx[token] = idx + 1
        idx2token[idx + 1] = token
    return token2idx, idx2token


def clean_report_iu_xray(report):
    report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
        .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
        .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                    replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report


def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

'''
#image_dir = "/home/ywu10/Documents/R2Gen/data/mimic_cxr/images"
image_dir = "/home/ywu10/Documents/R2Gen/data/iu_xray/images"
#ann_path = "/home/ywu10/Documents/R2Gen/data/mimic_cxr/annotation_label.json"
ann_path = "/home/ywu10/Documents/R2Gen/data/iu_xray/annotation_label.json"
ann = json.loads(open(ann_path, 'r').read())
ann = ann['test']
'''

prediction = pd.read_csv('/home/ywu10/vilmedic/vilmedic_iu_xray.csv')['prediction'].values.tolist()
ann_path = "/home/ywu10/Documents/R2Gen/data/iu_xray/annotation_label.json"
ann = json.loads(open(ann_path, 'r').read())
token2idx, idx2token = create_vocabulary(ann,clean_report_iu_xray,3)
ann = ann['test']
ground_truth = [clean_report_iu_xray(i['report']) for i in ann]
label =  [i['label'] for i in ann]
prediction = [clean_report_iu_xray(i) for i in prediction]

'''
imbalanced_eval(token2idx,prediction,ground_truth ,8)
imbalanced_eval(token2idx,prediction,ground_truth ,6)
imbalanced_eval(token2idx,prediction,ground_truth ,4)
'''
#normal
'''
ground_truth1 = [ground_truth[i] for i in range(len(label)) if label[i]==0]
prediction1 = [prediction[i] for i in range(len(label)) if label[i]==0]
test_met = compute_scores({i: [gt] for i, gt in enumerate(ground_truth1)},{i: [re] for i, re in enumerate(prediction1)})
print(f'normal:{test_met}')

ground_truth2 = [ground_truth[i] for i in range(len(label)) if label[i]==1]
prediction2 = [prediction[i] for i in range(len(label)) if label[i]==1]
test_met = compute_scores({i: [gt] for i, gt in enumerate(ground_truth2)},{i: [re] for i, re in enumerate(prediction2)})
print(f'abnormal:{test_met}')
'''

print('mimic')
prediction = pd.read_csv('/home/ywu10/vilmedic/vilmedic_mimic.csv')['prediction'].values.tolist()
ann_path = "/home/ywu10/Documents/R2Gen/data/mimic_cxr/annotation_label.json"
ann = json.loads(open(ann_path, 'r').read())
label =  [i['label'] for i in ann['test']]
token2idx, idx2token = create_vocabulary(ann,clean_report_mimic_cxr,10)
ann = ann['test']
ground_truth = [clean_report_mimic_cxr(i['report']) for i in ann]
prediction = [clean_report_mimic_cxr(i) for i in prediction]

'''
imbalanced_eval(token2idx,prediction,ground_truth,8)
imbalanced_eval(token2idx,prediction,ground_truth ,6)
imbalanced_eval(token2idx,prediction,ground_truth ,4)
'''

#normal
ground_truth1 = [ground_truth[i] for i in range(len(label)) if label[i][0]==1]
prediction1 = [prediction[i] for i in range(len(label)) if label[i][0]==1]
test_met = compute_scores({i: [gt] for i, gt in enumerate(ground_truth1)},{i: [re] for i, re in enumerate(prediction1)})
print(f'normal:{test_met}')

ground_truth2 = [ground_truth[i] for i in range(len(label)) if label[i][0]!=1]
prediction2 = [prediction[i] for i in range(len(label)) if label[i][0]!=1]
test_met = compute_scores({i: [gt] for i, gt in enumerate(ground_truth2)},{i: [re] for i, re in enumerate(prediction2)})
print(f'abnormal:{test_met}')






