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
import pickle
import csv
import progressbar
from multiprocessing import Pool
from modules.metrics import compute_b4scores


class evaluate_thread:
    def __init__(self,F,idx,dist,train_dataloader2):
        self.f = F
        self.idx = idx
        self.dist = dist
        self.train_dataloader2 = train_dataloader2

    def evaluate(self,i):
        index = self.train_dataloader2.dataset.img_id.index(self.idx[i])
        self.train_dataloader2.dataset.examples[index]['BLEU1'] = self.f[i]
        self.train_dataloader2.dataset.examples[index]['label_dist'] = self.dist[i]


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.bert_tokenizer = BertTokenizer.from_pretrained("/home/ywu10/Documents/r2genbaseline/classification")
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        self.classificationmodel = ClassificationModel().to(self.device)
        pretrained_params = torch.load('/home/ywu10/Documents/r2genbaseline/classification/best_classifier.pkl')
        self.classificationmodel.load_state_dict(pretrained_params, strict=False)
        for _,v in self.classificationmodel.named_parameters():
            v.requires_grad=False

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.compute_b4 = compute_b4scores

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
        self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        self.pool = Pool(os.cpu_count())

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):

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
        filename = os.path.join(self.checkpoint_dir, 'transformer2_current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'transformer2_model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self,resume_path='/home/ywu10/Documents/r2genbaseline/'):# resume_path='/hdd18t/models/model_iu_xray.pth'):
        resume_path = str(resume_path)
        resume_path = '/home/ywu10/Documents/r2genbaseline/model_iu_xray.pth'
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
                 test_dataloader,train_dataloader2):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.train_dataloader2 = train_dataloader2
        self.loss_func = torch.nn.KLDivLoss()


    def pretrain_epoch(self, epoch):
        train_loss = 0
        nor_ratio = 0
        ad_ratio = 0
        total = 0
        self.model.train()
        '''
        for batch_idx, (images_id, images, reports_ids, reports_masks,bert_reports_ids, bert_reports_masks, label) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks,label, bert_reports_ids, bert_reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device), label.to(self.device),bert_reports_ids.to(self.device), bert_reports_masks.to(self.device)
        '''
        for batch_idx, (images_id, images, reports_ids, reports_masks,_,_, label) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks,label = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device), label.to(self.device)

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

            #output,token_dist,_, weight = self.model(images, reports_ids, mode='train')
            output, _, = self.model(images, reports_ids, mode='train')
            '''
            pre_reports = torch.argmax(output,dim=-1)
            pre_reports = self.model.tokenizer.decode_batch(pre_reports.cpu().numpy())
            pre_reports,pre_mask = self.bert_encode(pre_reports)
            pre_label = self.classificationmodel(pre_reports,mask=pre_mask)
            pre_label2 = self.classificationmodel(bert_reports_ids,mask=bert_reports_masks)
            weight = F.softmax(torch.abs(pre_label - pre_label2),dim=0)*len(pre_label)
            weight = weight.detach()
            '''
            loss = self.criterion(output, reports_ids, reports_masks,weight=None)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        print(f'nor_ratio:{nor_ratio/total}, ad_ratio:{ad_ratio/total}')

    def evaluate(self):

        self.model.eval()
        p = progressbar.ProgressBar()
        p.start(len(self.train_dataloader))
        j = 0
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks, bert_reports_ids, bert_reports_masks,label) in p(enumerate(self.train_dataloader)):
                images, reports_ids, reports_masks,bert_reports_ids, bert_reports_masks,label = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                    self.device), bert_reports_ids.to(self.device), bert_reports_masks.to(self.device),label.to(self.device)

                output,_,_ = self.model(images,reports_ids,  mode='train')
                output = torch.argmax(output,dim=-1)
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                num_cls = max(torch.max(output),torch.max(reports_ids))+1
                output = (torch.sum(F.one_hot(output,num_cls),dim=1)[:,1:]).float()
                reports_ids = reports_ids[:, 1:]
                reports_ids_one_hot = (torch.sum(F.one_hot(reports_ids,num_cls),dim=1)[:,1:]>0).float()
                right = torch.sum(output * reports_ids_one_hot,dim=-1)
                precision = right/torch.sum(output,dim=-1)
                recall = right/torch.sum(reports_ids_one_hot,dim=-1)
                f = 2*(precision*(recall))/(precision+recall)
                f = f.cpu().numpy()
                pre_bert_ids, pre_bert_ids_mask = self.bert_encode(reports)
                pre_label = self.classificationmodel(pre_bert_ids,mask=pre_bert_ids_mask)
                dist = torch.abs(pre_label-label).sum(-1).cpu().numpy()
                #def evaluate_thread_fun(i):
                for i in range(len(reports)):
                    index = self.train_dataloader2.dataset.img_id.index(images_id[i])
                    self.train_dataloader2.dataset.examples[index]['BLEU1'] = f[i]
                    self.train_dataloader2.dataset.examples[index]['label_dist'] = dist[i]

                #evaluate_thread_fun = evaluate_thread(f,images_id,dist,self.train_dataloader2)
                #self.pool.map(evaluate_thread_fun,np.arange(len(reports)))
                #self.train_dataloader2 = evaluate_thread_fun.train_dataloader2
                '''
                for i in range(len(reports)):
                    index = self.train_dataloader2.dataset.img_id.index(images_id[i])
                    right = 0
                    recall = 0
                    precision = 0
                    a = ground_truths[i].split()
                    b = reports[i].split()
                    right += len([j for j in a if j in b])
                    recall += len(a)
                    precision += len(b)
                    recall = right/recall
                    precision = right/precision
                    f1 = 2*((recall*precision)/(recall+precision))
                    self.train_dataloader2.dataset.examples[index]['BLEU1'] = f1
                    self.train_dataloader2.dataset.examples[index]['label_dist'] = dist[i]
                '''
                j += 1
                p.update(j)
            p.finish()

        self.train_dataloader2.dataset.cal_weight()

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

        #self.pretrain_epoch(epoch)
        #task1_image_id,task1_image, task1_targets, task1_targets_masks,

        self.model.train()
        print(f'evaluation starting....')
        self.evaluate()
        print(f'training starting....')
        p = progressbar.ProgressBar()
        p.start(len(self.train_dataloader2))
        j = 0
        for batch_idx, (task1_image_id,task1_image, task1_targets, task1_targets_masks,\
                        task2_image_id,task2_image, task2_targets, task2_targets_masks, \
                        task2_bert_targets, task2_bert_targets_masks, task2_label) in p(enumerate(self.train_dataloader2)):

            #task 1

            task1_image, task1_targets, task1_targets_masks = task1_image.to(self.device), task1_targets.to(self.device), task1_targets_masks.to(self.device)
            output,token_dist, pre_label = self.model(task1_image, task1_targets, mode='train')
            reports_ids_one_hot = F.one_hot(task1_targets,output.size()[-1])
            reports_ids_one_hot = (reports_ids_one_hot>0).float()
            reports_ids_dist = F.log_softmax(torch.sum(reports_ids_one_hot,dim=1)/torch.sum(task1_targets_masks),dim=-1)
            #loss = F.kl_div(F.softmax(token_dist),F.softmax(reports_ids_dist))
            loss = F.kl_div(F.softmax(torch.mean(output,dim=1)),F.softmax(reports_ids_dist),reduction='batchmean')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
 

            task2_image, task2_targets, task2_targets_masks, task2_bert_targets, task2_bert_targets_masks, task2_label = \
                task2_image.to(self.device), task2_targets.to(self.device), task2_targets_masks.to(self.device), \
                        task2_bert_targets.to(self.device), task2_bert_targets_masks.to(self.device), task2_label.to(self.device)


            output,token_dist,_ = self.model(task2_image, task2_targets, mode='train')
        
            #pre_reports = torch.argmax(output,dim=-1)
            #pre_reports = self.model.tokenizer.decode_batch(pre_reports.cpu().numpy())
            #pre_reports,pre_mask = self.bert_encode(pre_reports)
            #pre_label = self.classificationmodel(pre_reports,mask=pre_mask)
            #pre_label2 = self.classificationmodel(task2_bert_targets,mask=task2_bert_targets_masks)
            #weight = F.softmax(torch.abs(pre_label - pre_label2),dim=0)*len(pre_label)
            #weight = weight.detach()
      
            loss = self.criterion(output, task2_targets, task2_targets_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            j += 1
            p.update(j)
        p.finish()

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        #print(f'nor_ratio:{nor_ratio/total}, ad_ratio:{ad_ratio/total}')


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
            label_list = []
            for batch_idx, (images_id, images, reports_ids, reports_masks,_,_, label) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                nor = [ground_truths[i] for i in range(len(label)) if label[i][0] == 1]
                ad = [ground_truths[i] for i in range(len(label)) if label[i][0] == 0]
                nor_pre = [reports[i] for i in range(len(label)) if label[i][0] == 1]
                ad_pre = [reports[i] for i in range(len(label)) if label[i][0] == 0]
                test_res.extend(nor)
                test_gts.extend(ad)
                label_list.append(label)
                test_res_pre.extend(nor_pre)
                test_gts_pre.extend(ad_pre)
                test_res1.extend(reports)
                test_gts1.extend(ground_truths)

            label_list = torch.cat(label_list,dim=0).to(self.device)


            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_res)},
                                    {i: [re] for i, re in enumerate(test_res_pre)})
            print(f'nor:{test_met}')
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_gts_pre)})

            print(f'ad:{test_met}')
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_res1)},
                                        {i: [re] for i, re in enumerate(test_gts1)})
            print(f'overall:{test_met}')

            log.update(**{'test_' + k: v for k, v in test_met.items()})
            self.imbalanced_eval(test_res1,test_gts1,8)
            self.imbalanced_eval(test_res1,test_gts1,6)
            self.imbalanced_eval(test_res1,test_gts1,4)

            #self.clinical_acc_sep(test_res_pre,test_gts_pre)
            #self.clinical_acc(test_res1,label_list)
            '''
            test_res1 = [i.replace('\n', '') for i in test_res1]
            test_res1 = [[i.replace(' .', '.')] for i in test_res1]
            file_name = "/home/ywu10/Documents/r2genbaseline/results/mimic_cxr_my_pre_report.csv"
            with open(file_name,"w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(test_res1)

            label_list = label_list.cpu().numpy()
            file = open('/home/ywu10/Documents/r2genbaseline/results/mimic_cxr_my_report.pkl', 'wb')
            pickle.dump(label_list, file)
            file.close()
            '''


        self.lr_scheduler.step()
        print(f'predict:{test_res1[-3:-1]}')
        print(f'ground_truth:{test_gts1[-3:-1]}')

        return log

    def clinical_acc_sep(self,prediction_nor,prediction_ad):
        pre_index, pre_mask = self.bert_encode(prediction_nor)
        pre_label = (self.classificationmodel(pre_index,mask=pre_mask).squeeze(-1)>0.5).float()
        nor_acc = torch.sum(pre_label == 0.)/len(pre_label)
        print(f'nor_acc is:{nor_acc}')

        pre_index, pre_mask = self.bert_encode(prediction_ad)
        pre_label = (self.classificationmodel(pre_index,mask=pre_mask).squeeze(-1)>0.5).float()
        nor_acc = torch.sum(pre_label == 1.)/len(pre_label)
        print(f'ad_acc is:{nor_acc}')


    def clinical_acc(self,prediction, label):

        pre_index, pre_mask = self.bert_encode(prediction)
        pre_label = (self.classificationmodel(pre_index,mask=pre_mask).squeeze(-1)>0.5).float()
        right = torch.sum(pre_label * label)
        precision =  right/torch.sum(pre_label)
        recall = right/torch.sum(label)
        F = 2*precision*recall/(precision+recall)
        print(f'precision:{precision},recall:{recall},F:{F}')

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
        r = np.array(right_)/np.array(recall_)
        print(f'recall:{r}')
        p = np.array(right_)/np.array(precision_)
        print(f'precision:{p}')
        print(precision_)
        print(recall_)
        print(f'f1:{2*r*p/(r+p)}')

    def imbalanced_eval2(self,pre,tgt,n):

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
