## Repo for 'MK News-video Recommendation' (sub-project of AI-HUB PJT)

[ Summary ]
- Baseline model v1: 뉴스 스크립트 (텍스트) 만을 활용한 모델.
 각 기사 스크립트를 TF-IDF 벡터로 표현 후, 이들 벡터 간 코사인 유사도 (Cosine similarity) 를 바탕으로 각 동영상 컨텐츠(기사) 간 유사도 계산하여 랭킹 부여
- Baseline model v2: 뉴스 스크립트 (텍스트) 외, 보도 일자 간 인접성을 추가로 고려한 모델. 
 각 동영상 컨텐츠(기사) 간 유사도 점수 (1) 에 추가적으로 보도 일자 간 인접성 점수(페널티) (2) 를 더하여 랭킹 부여
 
- 모델 관련 보다 상세한 설명은 '추천모델_설명_1015.pdf' 참고
