import pandas as pd
import torch
import numpy as np
import pickle
from modules.metrics import compute_b4scores
import progressbar

pre_report = pd.read_csv('/home/ywu10/Documents/r2genbaseline/results/mimic_cxr_cmn_pre_report.csv').values
report = pd.read_csv('/home/ywu10/Documents/SOTAgeneration/mymodel_generation_mimic2.csv')['truth'].values[1:]
blue4 = []
p = progressbar.ProgressBar()
p.start(len(report))
j = 0
for i in range(len(report)):
    #test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
    #                        {i: [re] for i, re in enumerate(test_res)})
    test_met = compute_b4scores({0: [report[i]]},
                                {0: pre_report[i].tolist()})
    blue4.append(test_met['BLEU_4'])
    j += 1
    p.update(j)
p.finish()

pd_data = {'prediction':[i[0] for i in pre_report],'bleu4':blue4}
pd_data = pd.DataFrame(pd_data)
pd_data.to_csv("baseline_generation_mimic.csv")
