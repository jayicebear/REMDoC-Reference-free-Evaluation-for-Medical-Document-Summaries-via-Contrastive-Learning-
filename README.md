## REMDoC-Reference-free-Evaluation-for-Medical-Document-Summaries-via-Contrastive-Learning

 **IEEE access accepted** <br/> 
[paper_link](https://ieeexplore.ieee.org/document/10804787/authors#authors) <br/> 
This is the official repository of REMDoC: Reference-free Evaluation for Medical Document Summaries via Contrastive Learning.

![image](https://github.com/user-attachments/assets/6ab5f013-ac07-40f4-9b3b-c843d87c8db1)

## 프로젝트 개요
이 프로젝트는 논문 *"REMDoC: Reference-Free Evaluation for Medical Document Summaries via Contrastive Learning"*에서 제안된 참조-free 의료 문서 요약 평가 메트릭 REMDoC를 구현한 코드입니다.  
RoBERTa-large 기반으로 Contrastive Learning을 사용해 참조 요약 없이 의료 문서 요약의 품질을 평가합니다.  
데이터 증강 기법(동의어 교체, Paraphrasing, 랜덤 삭제 등)을 통해 긍정/부정 쌍을 생성하고, 모델이 인간 판단과 유사한 점수를 출력하도록 학습합니다.  
목적: 기존 메트릭(ROUGE, BERTScore 등)의 한계를 극복하고, 의료 전문가 수준의 평가를 자동화.  
데이터셋: MSLR Cochrane (3,725개 요약 → 22,350개 증강 쌍).  
성능: 인간 평가와 상관계수 0.653 (기존 메트릭 대비 우수).

## 논문 요약
- **저자**: Jimin Lee, Ingeol Baek, Hwanhee Lee (Chung-Ang University)  
- **주요 기여**:  
  1. 참조-free 의료 요약 평가 메트릭 제안 (Contrastive Learning + 의료 특화 증강).  
  2. 기존 메트릭의 낮은 인간 상관성 문제 해결 (ROUGE: 0.053 미만 → REMDoC: 0.653).  
  3. 의료 문서의 미묘한 뉘앙스(용어, 약어, 맥락)를 포착.  
- **데이터 통계**: 원본 요약 3,725개, 증강 후 22,350개 쌍 (긍정: SR, PAR / 부정: RD, RS, AR, NER Swap).  
- **모델 성능**: Spearman 상관계수 0.653(의료 요약 인간 평가 데이터셋 기준).
