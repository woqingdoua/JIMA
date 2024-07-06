import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
from tqdm import tqdm
import sklearn

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    # /home/ywu10/Documents/R2Gen/data/iu_xray/images
    parser.add_argument('--image_dir', type=str, default='/home/ywu10/Documents/R2Gen/data/iu_xray/images', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='/home/ywu10/Documents/R2Gen/data/iu_xray/annotation_label.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')
    args = parser.parse_args()
    return args


class BaseDataset(Dataset):
    def __init__(self, args, split):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained("/home/ywu10/Documents/r2genbaseline/classification")
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = self.tokenizer.encode(self.examples[i]['report'])
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

            '''
            sentences = self.examples[i]['report'].split('. ')
            sentences_list = []
            for ss in sentences:
            '''
    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        label = example['label']
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (report_ids, report_masks, seq_length, label)
        return sample

class SampleDataLoader(DataLoader):
    def __init__(self, args, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.split)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        reports_ids, reports_masks, seq_lengths, label = zip(*data)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return torch.LongTensor(targets), torch.FloatTensor(targets_masks),torch.FloatTensor(label)


class ClassificationModel(nn.Module):
    def __init__(self,):
        super(ClassificationModel, self).__init__()

        self.model = BertForSequenceClassification.from_pretrained("/hdd18t/yuexin/Documents/r2genbaseline/classification",num_labels=158)
        '''
        for _,v in self.model.named_parameters():
            v.requires_grad=False
        '''
        #pretrained_params = torch.load('/hdd18t/yuexin/Documents/r2genbaseline/best_classifier_mimic.pkl')
        #self.model.load_state_dict(pretrained_params, strict=False)

        self.cls = nn.Linear(158,14)


    def forward(self,x,mask):

        x = self.model(x,attention_mask = mask)[0]
        pre_label = torch.sigmoid(self.cls(x))
        return pre_label


def train():
    args = parse_agrs()
    device = 1
    train_dataloader = SampleDataLoader(args,  split='train', shuffle=True)
    #val_dataloader = SampleDataLoader(args, split='val', shuffle=False)
    test_dataloader = SampleDataLoader(args, split='test', shuffle=False)
    classificationmodel = ClassificationModel().cuda(device)
    optimizer = torch.optim.Adam(classificationmodel.parameters(), lr=4e-5)
    best_loss = 10000

    pretrained_params = torch.load('/home/ywu10/Documents/r2genbaseline/classification/best_classifier_mimic.pkl')
    classificationmodel.load_state_dict(pretrained_params, strict=False)

    best_accuracy = 0.
    for epoch in range(50):


        train_loss = 0
        '''
        for batch_idx, (reports_ids, reports_masks, label) in tqdm(enumerate(train_dataloader)):
             reports_ids, reports_masks,label =  reports_ids.to(device), reports_masks.to(device), label.to(device)
             pre_label = classificationmodel(reports_ids, reports_masks)
             loss = F.binary_cross_entropy(pre_label,label)
             optimizer.zero_grad()
             loss.backward()
             train_loss += loss.item()
             optimizer.step()

        loss = train_loss/len(train_dataloader)
        print(f'train loss:{loss}')
        '''


        test_loss = 0
        classificationmodel.eval()
        correct=0
        total  = 0
        pre_labels = []
        ground_truth = []
        with torch.no_grad():
            for batch_idx, (reports_ids, reports_masks, label) in enumerate(test_dataloader):

                reports_ids = reports_ids[:,:512]
                reports_masks = reports_masks[:,:512]
                reports_ids, reports_masks,label =  reports_ids.to(device), reports_masks.to(device), label.to(device)
                pre_label = classificationmodel(reports_ids, reports_masks)
                pre_label = (pre_label.cpu()>0.5).float().tolist()
                label = label.float().tolist()
                pre_labels.extend(pre_label)
                ground_truth.extend(label)

        F1 = sklearn.metrics.f1_score(ground_truth,pre_labels,average='micro')

        print(f'test loss:{test_loss/len(test_dataloader)}')
        print(f'F1:{F1}')
        if best_accuracy < F1:
            best_accuracy = F1
            torch.save(classificationmodel.state_dict(),'/home/ywu10/Documents/r2genbaseline/classification/best_classifier_mimic.pkl')

    print(f'best accuracy:{best_accuracy}')


#train()