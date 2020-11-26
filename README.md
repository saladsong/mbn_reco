## Repo for 'MK News-video Recommendation' (sub-project of AI-HUB PJT)

[ Summary ]
- Baseline model v1 (Reco_v11): 뉴스 스크립트 (텍스트) 만을 활용한 모델.
 : 각 기사 스크립트를 TF-IDF 벡터로 표현 후, 이들 벡터 간 코사인 유사도 (Cosine similarity) 를 바탕으로 각 동영상 컨텐츠(기사) 간 유사도 계산하여 랭킹 부여.
- Baseline model v2 (Reco_v12) : 뉴스 스크립트 (텍스트) 외, 보도 일자 간 인접성을 추가로 고려한 모델. 
 : 각 동영상 컨텐츠(기사) 간 유사도 점수 (1) 에 추가적으로 보도 일자 간 인접성 점수(페널티) (2) 를 더하여 랭킹 부여.
 
 (베이스라인 모델 관련 보다 상세한 설명은 '추천모델_설명_201015.pdf' 참고)

-- 동영상 레이블 정보(visual features) 까지 활용한 모델은 기본적으로 Baseline v2 기반으로 보도 일자 간 인접성까지 고려하였음.
- Advanced model v1 (Reco_v21): 뉴스 동영상의 레이블 정보(visual features) 를 TF-IDF 기법으로 임베딩한 모델.
  : 각 동영상의 레이블 정보(visual features) 를 TF-IDF 벡터로 표현 후, 이들 벡터 간 코사인 유사도 (Cosine similarity) 를 바탕으로 각 동영상 컨텐츠(기사) 간 유사도 계산하여 랭킹 부여.
- Advanced model v2 (Reco_v22): 뉴스 동영상의 레이블 정보(visual features) 를 Word2Vec 기법으로 임베딩한 모델.
  : 각 동영상의 레이블 정보(visual features) 를 레이블 시퀀스로 간주하고 Word2Vec 알고리즘으로 학습함으로써 각 레이블에 대한 임베딩 획득.
  : 이를 바탕으로 각 동영상에 대한 임베딩 벡터를 획득하여, 이들 벡터 간 코사인 유사도 (Cosine similarity) 를 바탕으로 각 동영상 컨텐츠(기사) 간 유사도 계산하여 랭킹 부여.
  
-- 아래 모델은 11/30 추가 upload 예정
1) Advanced model v3 (Reco_v23): 뉴스 동영상의 레이블 정보(visual features) 를 LDA 기법으로 임베딩한 모델.
  : 각 동영상에 해당하는 레이블 셋을 하나의 문서로 간주하고 LDA 알고리즘을 통해 각 동영상에 대한 임베딩 획득하여, 이들 벡터 간 코사인 유사도 (Cosine similarity) 를 바탕으로 각 동영상 컨텐츠(기사) 간 유사도 계산하여 랭킹 부여.
2) Advanced model v4~v6 (Reco_v31~33): Reco_v21~23 각각에 뉴스 스크립트(텍스트) 데이터 기반 유사도를 추가로 반영한 모델
