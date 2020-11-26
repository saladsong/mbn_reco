import argparse
import numpy as np
import pandas as pd
from collections import Counter

import pickle

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel


## 전처리한 레이블 데이터 load & 레이블 시퀀스로 convert
def _get_label_seqs():
    with open('./vid_lab_dict.pkl', mode='rb') as f:
        vid_lab_dict = pickle.load(f)
        
    seqs_lst = list()
    all_label_lst = list()
    vid_idx_dict = dict()
    idx_vid_dict = dict()

    for idx, vid in enumerate(vid_lab_dict.keys()):
        seqs_lst.append( set(vid_lab_dict[vid]) )
        all_label_lst.extend( vid_lab_dict[vid] )
        vid_idx_dict[vid] = idx
        idx_vid_dict[idx] = vid
    
    return seqs_lst, all_label_lst, vid_idx_dict, idx_vid_dict
    

### 이하 TF-IDF 기반 Recommender
### 각 label = word (token) 으로 간주하고 TF-IDF 계산

def _get_fin_vocab(vocab):
    fin_vocab = {}
    
    c_vocab = Counter(vocab)
    cnt = len(fin_vocab)
    for voca in c_vocab:
        if (c_vocab[voca] > 1):   ## 등장 횟수 =1인 레이블은 필터링 (1020개 레이블 중 271개 / 26.57%)
            fin_vocab[voca] = cnt
            cnt += 1
            
    return fin_vocab

def _tok_to_idx(tok_seq, fin_vocab):
    res_idx = []
    for tok in tok_seq:
        if tok in fin_vocab:
            res_idx.append(fin_vocab[tok])
        else:
            pass
        
    return res_idx

def _seq_to_idx(lab_seqs_lst, fin_vocab):
    seq_to_idx = []

    for lab_seq in lab_seqs_lst:
        res_idx = _tok_to_idx(lab_seq, fin_vocab)
        seq_to_idx.append(res_idx)
    
    return seq_to_idx

def _preproc_for_csr(seq_to_idx, from_idx=0):
    user_lst = []
    vocab_lst = []

    for idx, seqs in enumerate(seq_to_idx[from_idx:]):
        uid = from_idx + idx
        uid_lst = []
        for tok in seqs:
            vocab_lst.append(tok)
            user_lst.append(uid)             

    return user_lst, vocab_lst

def _build_csr_mat(user_lst, vocab_lst, num_rows, num_cols):
    rows = np.array(user_lst)
    cols = np.array(vocab_lst)
    data = np.ones(len(user_lst))

    usr_itm_mat = sp.csr_matrix((data, (rows, cols)), shape=(num_rows, num_cols))
    
    return usr_itm_mat

def _build_tfidf_mat(usr_itm_mat):
    tfidf_trans = TfidfTransformer()
    tfidf_mat = tfidf_trans.fit_transform(usr_itm_mat)
    
    return tfidf_mat

def _artid_to_idx(data_econ, query_id):
    query_idx = data_econ[ data_econ['ART_ID'] == query_id ].index[0]

    return query_idx

def _idx_to_artid(data_econ, query_idx):
    if type(query_idx) == int:
        return data_econ['ART_ID'][query_idx]
    
    else:
        art_ids = list()
        for idx in query_idx:
            art_id = data_econ['ART_ID'][idx]
            art_ids.append(art_id)

    return art_ids

def _get_sim_mat(tfidf_mat):
    cos_sim = linear_kernel(tfidf_mat, tfidf_mat)
        
    return cos_sim


if __name__ == "__main__":
    ## data loading...
    seqs_lst, all_label_lst, vid_idx_dict, idx_vid_dict = _get_label_seqs()

    # build unique label dictionary & filter label_seqs based on final voca
    fin_vocab = _get_fin_vocab(all_label_lst)
    seq_to_idx = _seq_to_idx(seqs_lst, fin_vocab)
    num_rows, num_cols = len(seq_to_idx), len(fin_vocab)
    print('# of articles:', num_rows, 'init voca #', len(set(all_label_lst)), 'fin voca #:', num_cols)

    # build tf-idf matrix & get pre-calculated similarity matrix
    user_lst, vocab_lst = _preproc_for_csr(seq_to_idx)
    usr_itm_mat = _build_csr_mat(user_lst, vocab_lst, num_rows, num_cols)
    tfidf_mat = _build_tfidf_mat(usr_itm_mat)
    cos_sim_mat = _get_sim_mat(tfidf_mat)
    with open('adv_tfidf_cossim.pkl', 'wb') as f:
        pickle.dump(cos_sim_mat, f)

    # save the mapping btw video_id and idx
    with open('adv_lab_mapping_to_idx.pkl', 'wb') as f:
        pickle.dump(vid_idx_dict, f)
    with open('adv_lab_mapping_to_id.pkl', 'wb') as f:
        pickle.dump(idx_vid_dict, f)

    print('idx to vtid,', len(idx_vid_dict), idx_vid_dict[1],  'is same as 1190741')
    print('vid to idx,', len(vid_idx_dict), vid_idx_dict[1190741], 'is same as 1')