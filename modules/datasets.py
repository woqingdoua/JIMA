import os
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax
import torch.nn.functional as F
import progressbar

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None, example=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.bert_tokenizer = BertTokenizer.from_pretrained("/hdd18t/yuexin/Documents/r2genbaseline/classification")

        self.examples = self.ann[self.split]
        self.img_id = [i['id'] for i in self.examples]

        if example == None:
            p = progressbar.ProgressBar(len(self.examples))
            p.start()
            j = 0
            for i in range(len(self.examples)):
                self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
                self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

                self.examples[i]['bert_ids'] = self.bert_tokenizer.encode(self.examples[i]['report'])
                self.examples[i]['bert_mask'] = [1] * len(self.examples[i]['ids'])

                self.examples[i]['BLEU1'] = 0
                self.examples[i]['BLEU4'] = 0
                self.examples[i]['label_dist'] = 0
                j += 1
                p.update(j)
            p.finish()

        else:
            self.examples = example

    def __len__(self):
        return len(self.examples)


class IuxrayCurriculumDataset(BaseDataset):
    def cal_weight(self):

        self.normal_examples = [i for i in self.examples if i['label']==0]
        self.abnormal_examples = [i for i in self.examples if i['label']==1]
        self.weight1 = 0
        self.nor_weight2 = 0
        self.ab_weight2 = 0

        #nor_label_error = np.array([i['label_dist'] for i in self.normal_examples])
        #ab_label_error = np.array([i['label_dist'] for i in self.abnormal_examples])

        self.weight1 = np.array([i['BLEU1'] for i in self.examples])
        self.weight1 = F.gumbel_softmax(torch.tensor(max(self.weight1) - self.weight1),tau=5).numpy()

        label_error = np.array([i['label_dist'] for i in self.examples])
        self.weight2 = F.gumbel_softmax(torch.tensor(label_error),tau=5).numpy()

        self.weight3 = np.array([i['BLEU4'] for i in self.examples])
        self.weight3 = F.gumbel_softmax(torch.tensor(max(self.weight3) - self.weight3),tau=5).numpy()

        '''
        #self.ab_weight1 = np.array([i['BLEU1'] for i in self.abnormal_examples])
        #self.ab_weight1 = F.gumbel_softmax(torch.tensor(self.ab_weight1 - max(self.ab_weight1)),tau=3).numpy()

        self.nor_weight2 = np.array([i['BLEU4'] for i in self.normal_examples])
        self.nor_weight2 = F.gumbel_softmax(((F.gumbel_softmax(torch.tensor(self.nor_weight2 - max(self.nor_weight2)))  + F.gumbel_softmax(torch.tensor((nor_label_error))))/2),tau=3).numpy()
        self.ab_weight2 = np.array([i['BLEU4'] for i in self.abnormal_examples])
        '''

        #ratio = sum(nor_label_error)/(sum(nor_label_error) + sum(ab_label_error))
        #nor_amount = min(np.ceil((ratio) * len(self.examples)),len(self.normal_examples))

        #sampling
        self.task1_examples = np.random.choice(np.arange(len(self.examples)),size=len(self.examples),replace=True, p=self.weight1)
        self.task1_examples = [self.examples[i] for i in self.task1_examples]

        self.task2_examples = np.random.choice(np.arange(len(self.examples)),size=len(self.examples),replace=True, p=self.weight2)
        self.task2_examples = [self.examples[i] for i in self.task2_examples]

        self.task3_examples = np.random.choice(np.arange(len(self.examples)),size=len(self.examples),replace=True, p=self.weight3)
        self.task3_examples = [self.examples[i] for i in self.task3_examples]

        #self.task1_abnormal_examples = np.random.choice(np.arange(len(self.abnormal_examples)),size=ab_amount,replace=True,p=self.ab_weight1)

        #self.task1_abnormal_examples = [self.abnormal_examples[i] for i in self.task1_abnormal_examples]
        #task2_normal_examples = np.random.choice(np.arange(len(self.normal_examples)),size=int(nor_amount),replace=False)
        #task2_normal_examples = [self.normal_examples[i] for i in task2_normal_examples]
        #task2_abnormal_examples = np.random.choice(np.arange(len(self.abnormal_examples)),size=len(self.examples) - int(nor_amount),replace=False)
        #task2_abnormal_examples = [self.abnormal_examples[i] for i in task2_abnormal_examples]
        #self.task2_examples = task2_normal_examples + task2_abnormal_examples

    def __getitem__(self, idx):
        task1_example = self.task1_examples[idx]
        task2_example = self.task2_examples[idx]
        task1_label = task1_example['label']
        task1_image_path = task1_example['image_path']
        task1_image_1 = Image.open(os.path.join(self.image_dir, task1_image_path[0])).convert('RGB')
        task1_image_2 = Image.open(os.path.join(self.image_dir, task1_image_path[1])).convert('RGB')
        if self.transform is not None:
            task1_image_1 = self.transform(task1_image_1)
            task1_image_2 = self.transform(task1_image_2)
        task1_image = torch.stack((task1_image_1, task1_image_2), 0)
        task1_report_ids = task1_example['ids']
        task1_report_masks = task1_example['mask']
        task1_seq_length = len(task1_report_ids)
        task1_image_id = task1_example['id']

        task2_label = task2_example['label']
        task2_image_path = task2_example['image_path']
        task2_image_1 = Image.open(os.path.join(self.image_dir, task2_image_path[0])).convert('RGB')
        task2_image_2 = Image.open(os.path.join(self.image_dir, task2_image_path[1])).convert('RGB')
        if self.transform is not None:
            task2_image_1 = self.transform(task2_image_1)
            task2_image_2 = self.transform(task2_image_2)
        task2_image = torch.stack((task2_image_1, task2_image_2), 0)
        task2_report_ids = task2_example['ids']
        task2_report_masks = task2_example['mask']
        task2_bert_report_ids = task2_example['bert_ids']
        task2_bert_report_masks = task2_example['bert_mask']
        task2_seq_length = len(task2_report_ids)
        task2_image_id = task2_example['id']

        task3_example = self.task3_examples[idx]
        task3_label = task3_example['label']
        task3_image_path = task3_example['image_path']
        task3_image_1 = Image.open(os.path.join(self.image_dir, task3_image_path[0])).convert('RGB')
        task3_image_2 = Image.open(os.path.join(self.image_dir, task3_image_path[1])).convert('RGB')
        if self.transform is not None:
            task3_image_1 = self.transform(task3_image_1)
            task3_image_2 = self.transform(task3_image_2)
        task3_image = torch.stack((task3_image_1, task3_image_2), 0)
        task3_report_ids = task3_example['ids']
        task3_report_masks = task3_example['mask']
        task3_seq_length = len(task3_report_ids)
        task3_image_id = task3_example['id']

        sample = (task1_image_id,task1_image, task1_report_ids, task1_report_masks, task1_label,task1_seq_length, \
                  task2_image_id,task2_image, task2_report_ids, task2_report_masks,task2_bert_report_ids, \
                  task2_bert_report_masks, task2_seq_length, task2_label, \
                  task3_image_id,task3_image, task3_report_ids, task3_report_masks, task3_label,task3_seq_length )

        return sample


class MimicCurriculumDataset(BaseDataset):
    def cal_weight(self):

        self.normal_examples = [i for i in self.examples if i['label'][0]==1]
        self.abnormal_examples = [i for i in self.examples if i['label'][0]==0]
        self.weight1 = 0
        self.nor_weight2 = 0
        self.ab_weight2 = 0


        self.weight1 = np.array([i['BLEU1'] for i in self.examples])
        self.weight1[np.isnan(self.weight1)] = 0.
        self.weight1 = F.gumbel_softmax(torch.tensor(max(self.weight1) - self.weight1),tau=2).numpy()


        label_error = np.array([i['label_dist'] for i in self.examples])
        self.weight2 = F.gumbel_softmax(torch.tensor(label_error),tau=2).numpy()


        BLEU4 = np.array([i['BLEU4'] for i in self.examples])
        self.weight3 = F.gumbel_softmax(torch.tensor(max(BLEU4) - BLEU4),tau=2).numpy()


        #sampling
        self.task1_examples = np.random.choice(np.arange(len(self.examples)),size=len(self.examples),replace=True, p=self.weight1)
        self.task1_examples = [self.examples[i] for i in self.task1_examples]

        self.task2_examples = np.random.choice(np.arange(len(self.examples)),size=len(self.examples),replace=True, p=self.weight2)
        self.task2_examples = [self.examples[i] for i in self.task2_examples]

        self.task3_examples = np.random.choice(np.arange(len(self.examples)),size=len(self.examples),replace=True, p=self.weight3)
        self.task3_examples = [self.examples[i] for i in self.task3_examples]


    def __getitem__(self, idx):
        task1_example = self.task1_examples[idx]
        task2_example = self.task2_examples[idx]
        task1_label = task1_example['label']
        task1_image_path = task1_example['image_path']
        task1_image  = Image.open(os.path.join(self.image_dir,task1_image_path[0])).convert('RGB')
        if self.transform is not None:
            task1_image = self.transform(task1_image)
        task1_report_ids = task1_example['ids']
        task1_report_masks = task1_example['mask']
        task1_seq_length = len(task1_report_ids)
        task1_image_id = task1_example['id']

        task2_label = task2_example['label']
        task2_image_path = task2_example['image_path']
        task2_image  = Image.open(os.path.join(self.image_dir,task2_image_path[0])).convert('RGB')
        if self.transform is not None:
            task2_image = self.transform(task2_image)
        task2_report_ids = task2_example['ids']
        task2_report_masks = task2_example['mask']
        task2_bert_report_ids = task2_example['bert_ids']
        task2_bert_report_masks = task2_example['bert_mask']
        task2_seq_length = len(task2_report_ids)
        task2_image_id = task2_example['id']


        task3_example = self.task3_examples[idx]
        task3_label = task3_example['label']
        task3_image_path = task3_example['image_path']
        task3_image = Image.open(os.path.join(self.image_dir, task3_image_path[0])).convert('RGB')
        if self.transform is not None:
            task3_image = self.transform(task3_image)
        task3_report_ids = task3_example['ids']
        task3_report_masks = task3_example['mask']
        task3_seq_length = len(task3_report_ids)
        task3_image_id = task3_example['id']

        sample = (task1_image_id,task1_image, task1_report_ids, task1_report_masks, task1_label,task1_seq_length, \
                  task2_image_id,task2_image, task2_report_ids, task2_report_masks,task2_bert_report_ids, \
                  task2_bert_report_masks, task2_seq_length, task2_label, \
                  task3_image_id,task3_image, task3_report_ids, task3_report_masks, task3_label,task3_seq_length)

        return sample

class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        label = example['label']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        bert_report_ids = example['bert_ids']
        bert_report_masks = example['bert_mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks,bert_report_ids,bert_report_masks, seq_length, label)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        label = example['label']
        bert_report_ids = example['bert_ids']
        bert_report_masks = example['bert_mask']
        sample = (image_id, image, report_ids, report_masks,bert_report_ids,bert_report_masks,seq_length,label)
        return sample
