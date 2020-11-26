import argparse
import numpy as np
import pandas as pd
from collections import Counter

import pickle

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel


### 데이터 전처리 코드 생략 
### 이하 TF-IDF 기반 Recommender
### 전처리 / POS Tagging 완료된 데이터 (POSA Column) 이용해서 TF-IDF 계산

## 
def _posa_to_token(train_data):
    posa_to_token = []
    vocab = []
    tag_list = ['NNG', 'NNP', 'VV', 'VA', 'MAG', 'MAJ', 'MM', 'IC' ] ## to be modified

    for idx in range(len(train_data)):
        tok_lst = list()
        try:
            txt = train_data['ART_POS'][idx]
            pos_lst = txt.split(',')
            tok_lst = [ pos.split('/')[0].strip() for pos in pos_lst if pos.split('/')[-1] in tag_list ]
            vocab.extend(tok_lst)
            
        except:
            vocab.append('.')
            tok_lst.append('.')
            
        posa_to_token.append(tok_lst)
            
    return posa_to_token, vocab

def _get_fin_vocab(vocab):
    fin_vocab = {}
    
    c_vocab = Counter(vocab)
    cnt = len(fin_vocab)
    for voca in c_vocab:
        if (c_vocab[voca] < 5600) and (c_vocab[voca] > 1):
            fin_vocab[voca] = cnt
            cnt += 1
            
    return fin_vocab

def _tok_to_idx(article, fin_vocab):
    res_idx = []
    for tok in article:
        if tok in fin_vocab:
            res_idx.append(fin_vocab[tok])
        else:
            pass
        
    return res_idx

def _art_to_idx(txt_to_tok, fin_vocab):
    txt_to_idx = []

    for article in txt_to_tok:
        res_idx = _tok_to_idx(article, fin_vocab)
        txt_to_idx.append(res_idx)
    
    return txt_to_idx

def _preproc_for_csr(txt_to_idx, from_idx=0):
    user_lst = []
    vocab_lst = []

    for idx, title in enumerate(txt_to_idx[from_idx:]):
        uid = from_idx + idx
        uid_lst = []
        for tok in title:
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

def _mapping_artid_to_idx(data_econ):
    artid_to_idx = {}
    idx_to_artid = {}
    for idx in range(len(data_econ)):
        art_id = data_econ['ART_ID'][idx]
        artid_to_idx[art_id] = idx
        idx_to_artid[idx] = art_id

    return artid_to_idx, idx_to_artid

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

def _get_sim_article(tfidf_mat):
    cos_sim = linear_kernel(tfidf_mat, tfidf_mat)
        
    return cos_sim


if __name__ == "__main__":
    ## data loading...
    data_ = pd.read_csv('data_labeld_pos.csv', delimiter=';')

    # tokenize articles to token_lst -> (txt_to_tok), (vocab)
    txt_to_tok, vocab = _posa_to_token(data_)
    fin_vocab = _get_fin_vocab(vocab)
    num_rows, num_cols = len(txt_to_tok), len(fin_vocab)
    print('# of articles:', num_rows, 'init voca #', len(vocab), 'fin voca #:', num_cols)

    # filter token_lst based on final voca -> (txt_to_idx)
    txt_to_idx = _art_to_idx(txt_to_tok, fin_vocab)

    # build tf-idf matrix
    user_lst, vocab_lst = _preproc_for_csr(txt_to_idx)
    usr_itm_mat = _build_csr_mat(user_lst, vocab_lst, num_rows, num_cols)
    tfidf_mat = _build_tfidf_mat(usr_itm_mat)
    cos_sim_mat = _get_sim_article(tfidf_mat)
    with open('base_lab_cossim.pkl', 'wb') as f:
    #with open('base_cossim.pkl', 'wb') as f:
        pickle.dump(cos_sim_mat, f)

    # build mapping btw art_id and idx
    artid_to_idx, idx_to_artid = _mapping_artid_to_idx(data_)
    with open('base_lab_mapping_to_idx.pkl', 'wb') as f:
        pickle.dump(artid_to_idx, f)
    with open('base_lab_mapping_to_id.pkl', 'wb') as f:
        pickle.dump(idx_to_artid, f)

    print('idx to artid,', len(idx_to_artid), idx_to_artid[0],  'is same as 1120516')
    print('artid to idx,', len(artid_to_idx), artid_to_idx[1120520], 'is same as 1')