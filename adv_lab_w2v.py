import argparse
import numpy as np
import pandas as pd
from collections import Counter

import pickle

from gensim.models import Word2Vec
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel


## 레이블 데이터 load & 레이블 시퀀스로 convert
def _get_label_seqs():
    with open('./vid_lab_dict.pkl', mode='rb') as f:
        vid_lab_dict = pickle.load(f)
        
    seqs_lst = list()
    all_label_lst = list()
    vid_idx_dict = dict()
    idx_vid_dict = dict()

    for idx, vid in enumerate(vid_lab_dict.keys()):
        seqs_lst.append( vid_lab_dict[vid] ) 
        all_label_lst.extend( vid_lab_dict[vid] )
        vid_idx_dict[vid] = idx
        idx_vid_dict[idx] = vid
    
    return seqs_lst, all_label_lst, vid_idx_dict, idx_vid_dict
    

### 이하 Word2Vec 기반 Recommender
### label sequence 를 kinda word sequence 로 간주하고 W2V 적용하여 label embedding 학습

def _convert_seq(seq_data, max_len=50):
    res_seq = []
    pad_len = 0
    uniq_item = dict()
        
    for idx, seq in enumerate(seq_data):
        tmp_seq = []
        
        seq_len = len(seq)
        if seq_len < max_len:
                pad_len = max_len - seq_len
                seq = ['0']*pad_len + seq
                
        for label in seq[:max_len]:
            if label not in uniq_item:
                uniq_item[label] = 1
            tmp_seq.append(label)

        res_seq.append(tmp_seq)
    
    #print('num of seqs:', len(res_seq))
    #print('num of uniq labels:', len(uniq_item.keys()))
    
    return res_seq

def _get_w2v_emb(seq_data, hid_dim, window):  # 0 for CBOW, 1 for skip_gram
    wv_model = Word2Vec(sentences=seq_data, size=hid_dim, window=window, min_count=1, workers=4, sg=0)
    vocab = wv_model.wv.index2word
    print('pre-emb vocab size:' , len(vocab))
    
    return wv_model, vocab

def _get_seq_emb(voca, w2v_model, hid_dim=50):
    emb_matrix = np.zeros([len(voca), hid_dim])
    
    for idx, v in enumerate(voca):
        if v == '0':
            embed = np.zeros(hid_dim)
        else:
            try:
                embed = w2v_model.wv[v]
            except KeyError:
                print('keyError', idx, v, voca)
                break
        emb_matrix[idx] = embed
        
    return emb_matrix

def _make_emb_matrix(train_seqs, wv_model, hid_dim=50, max_len=50):
    emb_matrix = np.zeros([len(train_seqs), max_len, hid_dim])
    
    for idx, voca in enumerate(train_seqs):
        voca_emb = _get_seq_emb(voca, wv_model, hid_dim)
        emb_matrix[idx] = voca_emb
    
    return emb_matrix
    
def _get_sim_mat(emb_mat):
    cos_sim = linear_kernel(emb_mat, emb_mat)
        
    return cos_sim


if __name__ == "__main__":
    ## data loading...
    seqs_lst, all_label_lst, vid_idx_dict, idx_vid_dict = _get_label_seqs()
    
    ## make any label sequence being same length (zero-pad or crop)
    maxlen, hiddim = 50, 50
    train_seqs = _convert_seq(seqs_lst, max_len=maxlen)
    
    ## build w2v embeddings & convert label seq into embedding matrix of (max_len) x (hid_dim) 
    wv_model, vocab = _get_w2v_emb(seqs_lst, hid_dim=hiddim, window=5)
    print('# of articles:', len(train_seqs), 'init voca #', len(set(all_label_lst)), 'pre-embeded voca #:', len(vocab))
    
    emb_matrix = _make_emb_matrix(train_seqs, wv_model, hid_dim=hiddim, max_len=maxlen)
    emb_matrix_flatten = emb_matrix.reshape(len(train_seqs), maxlen*hiddim)
    
    cos_sim_mat = _get_sim_mat(emb_matrix_flatten)
    with open('adv_w2v_cossim.pkl', 'wb') as f:
        pickle.dump(cos_sim_mat, f)

    # save the mapping btw video_id and idx
    #with open('adv_lab_mapping_to_idx.pkl', 'wb') as f:
    #    pickle.dump(vid_idx_dict, f)
    #with open('adv_lab_mapping_to_id.pkl', 'wb') as f:
    #    pickle.dump(idx_vid_dict, f)

    #print('idx to vtid,', len(idx_vid_dict), idx_vid_dict[1],  'is same as 1190741')
    #print('vid to idx,', len(vid_idx_dict), vid_idx_dict[1190741], 'is same as 1')