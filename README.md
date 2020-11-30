## Repo for 'MK News-video Recommendation' (sub-project of AI-HUB PJT)

[ Summary ]

- Reco_v1.x : Baseline model
 1) Reco_v1.1 : 1차 베이스라인 모델. 뉴스 스크립트(text) 만을 활용한 모델. 각 기사 스크립트를 TF-IDF 벡터로 표현 후, 이들 벡터 간 코사인 유사도 (Cosine similarity) 를 바탕으로 각 동영상 컨텐츠(기사) 간 유사도 계산하여 랭킹 부여.
 2) Reco_v1.2 : 2차 베이스라인 모델. 뉴스 스크립트(text) 외, 보도 일자 간 인접성을 추가로 고려한 모델. 각 동영상 컨텐츠(기사) 간 유사도 점수 (1) 에 추가적으로 보도 일자 간 인접성 점수(페널티) (2) 를 더하여 랭킹 부여.
 
  (덧. 베이스라인 모델 관련 보다 상세한 설명은 '추천모델_설명_201015.pdf' 참고)

- Reco_v2.x : 1차 Advanced model. 동영상 레이블 정보(visual features) 활용 + 보도 일자 간 인접성 고려한 모델.
 1) Reco_v2.1 : 뉴스 동영상의 레이블 정보(visual features) 를 TF-IDF 기법으로 임베딩한 모델. 각 동영상의 레이블 정보(visual features) 를 TF-IDF 벡터로 표현 후, 이들 벡터 간 코사인 유사도 (Cosine similarity) 를 바탕으로 각 동영상 컨텐츠(기사) 간 유사도 계산하여 랭킹 부여.
 2) Reco_v2.2 : 뉴스 동영상의 레이블 정보(visual features) 를 Neural Word Embedding 기법의 하나인 Word2Vec 을 활용하여 임베딩한 모델. 각 동영상의 레이블 정보(visual features) 를 레이블 시퀀스로 간주하고 Word2Vec 알고리즘으로 학습함으로써 각 레이블에 대한 임베딩 획득. 이를 바탕으로 각 동영상에 대한 임베딩 벡터를 획득하여, 이들 벡터 간 코사인 유사도 (Cosine similarity) 를 바탕으로 각 동영상 컨텐츠(기사) 간 유사도 계산하여 랭킹 부여.
 3) Reco_v2.3 : 뉴스 동영상의 레이블 정보(visual features) 를 문서 토픽 모델링 기법의 하나인 LDA(Latent Dirichlet Allocation) 을 활용하여 임베딩한 모델. 각 동영상에 해당하는 labels set 를 하나의 문서(bag-of-words) 로 간주 -> 각 동영상을 구성하는 Topics = embedding features 로 보고 LDA 적용 (Num of Topics K = Embedding dimension D). 이를 바탕으로 동영상(문서) 단위 embedding 획득하여, 이들 벡터 간 코사인 유사도 (Cosine similarity) 를 바탕으로 각 동영상 컨텐츠(기사) 간 유사도 계산하여 랭킹 부여.
  
- Reco_v3.x : 2차 Advanced model. 뉴스 스크립트(text) + 동영상 레이블 정보(visual features) + 보도 일자 간 인접성 모두 고려한 Ensemble 모델. 각 Reco_v2.x 에 추가로 뉴스 스크립트(text) 기반 유사도를 추가로 고려한 모델로서, 스크립트 : 레이블 각각의 가중치는 0.8 : 0.2 를 default 값으로 설정.
