import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from def_train_eval import *

import pickle            


#DATA = 'OTH'
#SUFIX = '1stS15-5'
#train_seq_len = 5
#pred_seq_len = 20

DATA = 'ARGO'
SUFIX = '1stS1new'
train_seq_len = 20
pred_seq_len = 30

device = torch.device("cuda:0")
s2 = True
TRAIN = False
EVAL = True


DIR = '../resources/data/{}/'.format(DATA)
MODEL_DIR = '../resources/trained_models/'

epochs = 100

save_per_epochs = 5


if __name__ == "__main__":
    
    if TRAIN:
        if DATA == 'OTH':
            f1 = open ( DIR + 'stream1_obs_data_5_{}_train.pkl'.format(pred_seq_len), 'rb')  # 'r' for reading; can be omitted
            g1 = open ( DIR + 'stream1_pred_data_5_{}_train.pkl'.format(pred_seq_len), 'rb')  # 'r' for reading; can be omitted
        
        else:
            f1 = open ( DIR + 'stream1_obs_data_train.pkl', 'rb')  # 'r' for reading; can be omitted
            g1 = open ( DIR + 'stream1_pred_data_train.pkl', 'rb')  # 'r' for reading; can be omitted


        tr_seq_1 = pickle.load ( f1 )  # load file content as mydict
        pred_seq_1 = pickle.load ( g1 )  # load file content as mydict
        f1.close()
        g1.close()

        
        if s2:
            f2 = open ( DIR + 'stream2_obs_data_5_{}_train.pkl', 'rb')  # 'r' for reading; can be omitted
            g2 = open ( DIR + 'stream2_pred_data_5_{}_train.pkl', 'rb')  # 'r' for reading; can be omitted
            f3 = open ( DIR + 'stream2_obs_eigs_5_{}_train.pkl', 'rb')  # 'r' for reading; can be omitted
            g3 = open ( DIR + 'stream2_pred_eigs_5_{}_train.pkl', 'rb')  # 'r' for reading; can be omitted
            tr_seq_2 = pickle.load ( f2 )  # load file content as mydict
            pred_seq_2 = pickle.load ( g2 )  # load file content as mydict
            tr_eig_seq = pickle.load ( f3 )
            pred_eig_seq = pickle.load ( g3 )
            f2.close ()
            f3.close ()
            g2.close ()
            g3.close ()
        else:
            tr_seq_2 = []
            pred_seq_2 = []
            tr_eig_seq = []
            pred_eig_seq = []

        encoder1, decoder1 = trainIters(epochs, tr_seq_1 , pred_seq_1, tr_seq_2, pred_seq_2, tr_eig_seq, pred_eig_seq, DATA, SUFIX, s2, print_every=1, save_every=save_per_epochs)
    
    if EVAL:
        print('start evaluating {}{}...'.format(DATA, SUFIX))
        if DATA == 'OTH':
            f1 = open ( DIR + 'stream1_obs_data_5_{}_test.pkl'.format(pred_seq_len), 'rb')  # 'r' for reading; can be omitted
            g1 = open ( DIR + 'stream1_pred_data_5_{}_test.pkl'.format(pred_seq_len), 'rb')  # 'r' for reading; can be omitted
        else:
            f1 = open ( DIR + 'stream1_obs_data_test.pkl', 'rb')  # 'r' for reading; can be omitted
            g1 = open ( DIR + 'stream1_pred_data_test.pkl', 'rb')  # 'r' for reading; can be omitted
        tr_seq_1 = pickle.load ( f1 )  # load file content as mydict
        pred_seq_1 = pickle.load ( g1 )  # load file content as mydict
        f1.close ()

        g1.close ()



        for eval_iter in range(10):
            print('eval_iter {}'.format(eval_iter))
            eval(10, tr_seq_1, pred_seq_1, DATA, SUFIX)
