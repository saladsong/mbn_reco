import argparse
import numpy as np
import pandas as pd
from collections import Counter
import scipy.sparse as sp

import pickle

from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.callbacks import CoherenceMetric
from gensim.models.coherencemodel import CoherenceModel

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
    

### 이하 LDA(Latent Dirichlet Allocation) 기반 Recommender
### 각 동영상을 하나의 문서로 보고 corresponding labels 를 kinda bag-of-words 로 간주
### 각 동영상(문서) 를 구성하는 Topics = embedding features 로 보고 LDA 적용하여 동영상(문서) 단위 embedding 학습
### Num of Topics K = Embedding dimension D

## 전체 레이블 시퀀스(Corpus) 바탕으로 Dictionary 구축 & 등장 빈도가 너무 많거나 적은 레이블은 Filtering
def _build_dictionary(seqs_lst):
    dictionary = corpora.Dictionary(seqs_lst)

    dictionary.filter_extremes(no_below=2, no_above=0.05) ## 등장횟수 1회 or 전체 corpus 내 등장비율 0.05 이상인 label 제거 
    corpus = [dictionary.doc2bow(label_seq) for label_seq in seqs_lst] 

    print('Number of unique tokens (labels): %d' % len(dictionary))
    print('Number of documents (news video): %d' % len(corpus))

    return dictionary, corpus

## for hyper-parameter setting __ finding optimal # of topics 
def compute_coh_values(dictionary, corpus, texts, start, max_limit, step):
    coherence_values = []
    topic_num = []
    model_list = [] 

    for num_topics in range(start, max_limit, step): 
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics) 
        model_list.append(model) 
        cohmodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v') 
        coherence_values.append(cohmodel.get_coherence())
        topic_num.append(num_topics)
        
    return model_list, coherence_values, topic_num

def find_optimal_number_of_topics(dictionary, corpus, processed_data): 
    limit = 300
    start = 50 
    step = 50
    model_list, coh_values, topic_num = compute_coh_values(dictionary=dictionary, corpus=corpus, texts=processed_data, start=start, limit=limit, step=step) 
    
    max_coh_val = max(coh_values)
    max_coh_idx = coh_values.index(max_coh_val)
    max_topic_num = topic_num[max_coh_idx]

    #x = range(start, limit, step) 
    #plt.plot(x, coh_values) 
    #plt.xlabel("Num Topics") 
    #plt.ylabel("Coherence score") 
    #plt.legend(("coherence_values"), loc='best') 
    #plt.show()

    return max_topic_num, max_coh_val 

def _get_lda_model(seqs_lst, corpus, dictionary, k=200):
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k)  ## # of uniq. tokens = 661

    cohmodel = CoherenceModel(model=model, texts=seqs_lst, dictionary=dictionary, coherence='c_v') 
    print('coherence level:', cohmodel.get_coherence())

    return model

def _get_vid_embeddings(lda_model, corpus, article_num, k=200):
    emb_matrix = np.zeros([article_num, k], dtype=np.float32)
    all_topics_prob = lda_model.get_document_topics(corpus, minimum_probability=0, per_word_topics=False)
    
    for vid, prob_lst in enumerate(all_topics_prob):
        emb_lst = np.zeros(k)
        for prob in prob_lst:
            emb_lst[ prob[0] ] = prob[1]   # prob[0] == idx (0 ~ (k-1))
        emb_matrix[vid] = emb_lst
    
    return emb_matrix      
    
def _get_sim_mat(emb_mat):
    cos_sim = linear_kernel(emb_mat, emb_mat)
        
    return cos_sim


if __name__ == "__main__":
    ## data loading...
    seqs_lst, all_label_lst, vid_idx_dict, idx_vid_dict = _get_label_seqs()

    ## build label dictionary & corpus
    dictionary, corpus = _build_dictionary(seqs_lst)

    ## for finding optimal num of topics
    #max_topic_num, max_coh_val = find_optimal_number_of_topics(dictionary, corpus, processed_data)
    
    ## get LDA model & build embedding matrix
    article_num = len(corpus)
    num_topics = 200
    lda_model = _get_lda_model(seqs_lst, corpus, dictionary, k=num_topics)
    lda_emb_matrix = _get_vid_embeddings(lda_model, corpus, article_num, k=num_topics)
    print('# of articles:', lda_emb_matrix.shape[0])
    
    cos_sim_mat = _get_sim_mat(lda_emb_matrix)
    with open('adv_lda_cossim.pkl', 'wb') as f:
        pickle.dump(cos_sim_mat, f)

    # save the mapping btw video_id and idx
    #with open('adv_lab_mapping_to_idx.pkl', 'wb') as f:
    #    pickle.dump(vid_idx_dict, f)
    #with open('adv_lab_mapping_to_id.pkl', 'wb') as f:
    #    pickle.dump(idx_vid_dict, f)

    print('idx to vtid,', len(idx_vid_dict), idx_vid_dict[1],  'is same as 1190741')
    print('vid to idx,', len(vid_idx_dict), vid_idx_dict[1190741], 'is same as 1')