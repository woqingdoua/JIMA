import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import IuxrayMultiImageDataset, MimicCurriculumDataset,IuxrayCurriculumDataset,MimiccxrSingleImageDataset


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle,example=None):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
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
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform, example=example)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)


    '''
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks,seq_lengths, label = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)


        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks


        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), torch.FloatTensor(label)
    '''
    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks,bert_reports_ids, bert_reports_masks, seq_lengths, label = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)
        max_seq_length2 = max([len(i) for i in bert_reports_ids])

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        bert_targets = np.zeros((len(bert_reports_ids), max_seq_length2), dtype=int)
        bert_targets_masks = np.zeros((len(bert_reports_ids), max_seq_length2), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        for i, report_ids in enumerate(bert_reports_ids):
            bert_targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            bert_targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), \
               torch.LongTensor(bert_targets), torch.FloatTensor(bert_targets_masks), torch.FloatTensor(label)



class CurriculumDataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
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
            self.dataset = IuxrayCurriculumDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimicCurriculumDataset(self.args, self.tokenizer, self.split, transform=self.transform)

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
        task1_image_id,task1_image, task1_report_ids, task1_report_masks, task1_label,task1_seq_length, \
        task2_image_id,task2_image, task2_report_ids, task2_report_masks,task2_bert_report_ids, \
        task2_bert_report_masks, task2_seq_length, task2_label, \
        task3_image_id,task3_image, task3_report_ids, task3_report_masks, task3_label,task3_seq_length = zip(*data)


        task1_image = torch.stack(task1_image, 0)
        task1_max_seq_length = max(task1_seq_length)

        task1_targets = np.zeros((len(task1_report_ids), task1_max_seq_length), dtype=int)
        task1_targets_masks = np.zeros((len(task1_report_ids), task1_max_seq_length), dtype=int)

        for i, report_ids in enumerate(task1_report_ids):
            task1_targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(task1_report_masks):
            task1_targets_masks[i, :len(report_masks)] = report_masks

        task2_image = torch.stack(task2_image, 0)
        task2_max_seq_length = max(task2_seq_length)
        task2_max_seq_length2 = max([len(i) for i in task2_bert_report_ids])

        task2_targets = np.zeros((len(task2_report_ids), task2_max_seq_length), dtype=int)
        task2_targets_masks = np.zeros((len(task2_report_ids), task2_max_seq_length), dtype=int)

        task2_bert_targets = np.zeros((len(task2_bert_report_ids), task2_max_seq_length2), dtype=int)
        task2_bert_targets_masks = np.zeros((len(task2_bert_report_ids), task2_max_seq_length2), dtype=int)

        for i, report_ids in enumerate(task2_report_ids):
            task2_targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(task2_report_masks):
            task2_targets_masks[i, :len(report_masks)] = report_masks

        for i, report_ids in enumerate(task2_bert_report_ids):
            task2_bert_targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(task2_bert_targets_masks):
            task2_bert_targets_masks[i, :len(report_masks)] = report_masks


        task3_image = torch.stack(task3_image, 0)
        task3_max_seq_length = max(task3_seq_length)

        task3_targets = np.zeros((len(task3_report_ids), task3_max_seq_length), dtype=int)
        task3_targets_masks = np.zeros((len(task3_report_ids), task3_max_seq_length), dtype=int)

        for i, report_ids in enumerate(task3_report_ids):
            task3_targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(task3_report_masks):
            task3_targets_masks[i, :len(report_masks)] = report_masks


        return task1_image_id,task1_image, torch.LongTensor(task1_targets), torch.FloatTensor(task1_targets_masks), \
               task2_image_id,task2_image, torch.LongTensor(task2_targets), torch.FloatTensor(task2_targets_masks), \
               torch.LongTensor(task2_bert_targets), torch.FloatTensor(task2_bert_targets_masks), torch.FloatTensor(task2_label), \
               task3_image_id,task3_image, torch.LongTensor(task3_targets), torch.FloatTensor(task3_targets_masks)

