import os
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf
import numpy as np
import torch.nn.functional as F
from pretrained_classification_model import ClassificationModel
from transformers import BertTokenizer, BertForSequenceClassification


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.bert_tokenizer = BertTokenizer.from_pretrained("/home/ywu10/Documents/r2genbaseline/classification")
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        self.classificationmodel = ClassificationModel().to(self.device)
        for _,v in self.classificationmodel.named_parameters():
            v.requires_grad=False
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        #if args.resume is not None:
        #self._resume_checkpoint()

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):

            if epoch <0:
                self.pre_train_epoch()
                continue

            else:
                result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self,resume_path='/home/ywu10/Documents/R2GenCMN2/model_mimic_cxr.pth'):# resume_path='/hdd18t/models/model_iu_xray.pth'):
        resume_path = str(resume_path)
        #resume_path = str('/hdd18t/models/model_iu_xray.pth')
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.loss_func = torch.nn.KLDivLoss()


    def pre_train_epoch(self):

        self.model.train()
        train_loss = 0
        for batch_idx, (images_id, images, reports_ids, reports_masks, label) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks,label = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device), label.to(self.device)

            pre_label = self.model(images, reports_ids, mode='pretrain')
            loss = F.binary_cross_entropy(pre_label.squeeze(-1),label)
            self.optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        print(f'pretrain loss:{train_loss/len(self.train_dataloader)}')

    def bert_encode(self,x):

        xs = []
        xs_mask = []
        for i in range(len(x)):
            pre_reports = self.bert_tokenizer.encode(x[i])
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

        bert_targets, bert_targets_masks = torch.tensor(bert_targets).to(self.device), torch.tensor(bert_targets_masks).to(self.device)

        return bert_targets, bert_targets_masks

    def _train_epoch(self, epoch):
        train_loss = 0
        nor_ratio = 0
        ad_ratio = 0
        total = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks,bert_reports_ids, bert_reports_masks, label) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks,label, bert_reports_ids, bert_reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device), label.to(self.device),bert_reports_ids.to(self.device), bert_reports_masks.to(self.device)

            '''
            output,token_dist, pre_label, _ = self.model(images, reports_ids, mode='train')
            reports_ids_one_hot = F.one_hot(reports_ids,output.size()[-1])
            reports_ids_dist = F.log_softmax(torch.sum(reports_ids_one_hot,dim=1)/torch.sum(reports_masks),dim=-1)
            #loss = F.kl_div(token_dist,reports_ids_dist) + F.binary_cross_entropy(pre_label.squeeze(-1),label)
            loss = F.kl_div(F.softmax(token_dist),F.softmax(reports_ids_dist))
            self.optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            '''


            ad_ratio += torch.sum(label == 1.).item()
            nor_ratio += torch.sum(label == 0.).item()
            total += len(label)

            output,token_dist,_, weight = self.model(images, reports_ids, mode='train')
            pre_reports = torch.argmax(output,dim=-1)
            pre_reports = self.model.tokenizer.decode_batch(pre_reports.cpu().numpy())
            pre_reports,pre_mask = self.bert_encode(pre_reports)
            pre_label = self.classificationmodel(pre_reports,mask=pre_mask)
            pre_label2 = self.classificationmodel(bert_reports_ids,mask=bert_reports_masks)
            weight = F.softmax(torch.abs(pre_label - pre_label2),dim=0)*len(pre_label)
            weight = weight.detach()
            loss = self.criterion(output, reports_ids, reports_masks,weight=weight)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        print(f'nor_ratio:{nor_ratio/total}, ad_ratio:{ad_ratio/total}')

        self.model.eval()
        with torch.no_grad():   
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks,_,_, label) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            test_gts_pre, test_res_pre = [], []
            test_gts1, test_res1 = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks,_,_, label) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                nor = [ground_truths[i] for i in range(len(label)) if label[i] == 0]
                ad = [ground_truths[i] for i in range(len(label)) if label[i] == 1]
                nor_pre = [reports[i] for i in range(len(label)) if label[i] == 0]
                ad_pre = [reports[i] for i in range(len(label)) if label[i] == 1]
                test_res.extend(nor)
                test_gts.extend(ad)
                test_res_pre.extend(nor_pre)
                test_gts_pre.extend(ad_pre)
                test_res1.extend(reports)
                test_gts1.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_res)},
                                        {i: [re] for i, re in enumerate(test_res_pre)})
            print(f'nor:{test_met}')
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_gts_pre)})

            print(f'ad:{test_met}')
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_res1)},
                                        {i: [re] for i, re in enumerate(test_gts1)})

            log.update(**{'test_' + k: v for k, v in test_met.items()})
            self.imbalanced_eval(test_res1,test_gts1,8)
            self.imbalanced_eval(test_res1,test_gts1,6)
            self.imbalanced_eval(test_res1,test_gts1,4)


        self.lr_scheduler.step()
        print(f'predict:{test_res[-1]}')
        print(f'ground_truth:{test_gts[-1]}')


        return log

    def imbalanced_eval(self,pre,tgt,n):

        #words = dict(sorted(dict(self.model.tokenizer.counter).items(), key=lambda x: x[1]))
        words = [w for w in self.model.tokenizer.token2idx_][:-2]
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
        recall = np.array(right_)/np.array(recall_)
        print(f'recall:{recall}')
        precision = np.array(right_)/np.array(precision_)
        print(f'precision:{precision}')
        print(precision_)
        print(recall_)
        f1 = 2*((recall*precision)/(recall+precision))
        print(f'F1:{f1}')
