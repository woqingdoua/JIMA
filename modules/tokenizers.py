import json
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()
        self.token2idx_, self.idx2token_,self.vocab_order = self.create_vocabulary2()
        #self.data_analy()
        print('a')

    def data_analy(self):
        data = [self.vocab_order[i] for i in self.vocab_order if self.vocab_order[i] >=3]
        counts, bins = np.histogram(data,range=(3,max(data)),bins=8)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.show()
        data = [data[i] for i in range(len(data)-1,1,-1) if data[i]>=10]
        counts, bins = np.histogram(data,range=(10,max(data)),bins=30)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.show()
        print('a')
        '''
        max_num = self.vocab_order['.']
        gap = max_num//30
        y_axis = [0]*30
        for i in data:
            v = i//gap
            y_axis[v] += 1
        plt.hist(y_axis)
        plt.show()
        print('a')
        '''

    def create_vocabulary2(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocabs = dict(sorted(counter.items(),key = lambda x:x[1],reverse=True))
        vocab =  [i  for  i in vocabs if  vocabs[i]>=self.threshold]
        vocab.append('<unk>')
        #vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token

        '''
        mapping = list(vocab)[:len(vocab)//4]
        high_mappings = []
        for mm in mapping:
            high_mappings.append(self.token2idx[mm])

        mapping = list(vocab)[len(vocab)//2:]
        low_mappings =  []
        for mm in mapping:
            low_mappings.append(self.token2idx[mm])
  
        weight = []
        for i in self.token2idx:
            if i in mapping:
                weight.append(0.)
            else:
                weight.append(1.)

        weight.append(1.)
        '''
        vob_freq = []
        cc = [i for i in dict(counter) if dict(counter)[i]>=self.threshold]
        gap = sum([dict(counter)[i] for i in dict(counter) if dict(counter)[i]>=self.threshold])//3
        counter = dict(counter)
        for i in cc:
            vob_freq.extend([i]*counter[i])
        vob_freq2 = []
        for i in range(3):
            vob_freq2.append(list(set(vob_freq[i*gap:(i+1)*gap])))
        del vob_freq

        return token2idx, idx2token, vocabs

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report


    def clean_report_mimic_cxr(self, report):
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

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
