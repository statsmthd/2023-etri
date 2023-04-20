# 2023 제2회 ETRI 휴먼이해 인공지능 논문경진대회
>본 대회는 한국전자통신연구원(ETRI)이 주최하고 과학기술정보통신부와 국가과학기술연구회(NST)가 후원합니다.


- 멀티모달 감정 데이터셋 활용 감정인식 기술 분야 [바로가기](https://aifactory.space/competition/detail/2234)
- KEMDy19 (성우 대상 상황극) 데이터셋 [바로가기](https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR)

## Code
`audio CNN.py` : 음성 신호 감정표현 모델

`eda XGBClassifier bf.py` : EDA 감정반응+감정표현 모델

`eda XGBClassifier now.py` : EDA 감정표현 모델

`text KoBERT bf.py` : 텍스트 감정반응+감정표현 모델

`soft voting.py` : 최종 멀티모달


# 논문 주제
## Abstract
본 연구는 대화에서 텍스트, 생체신호(EDA), 음성신호 데이터와 더불어 감정반응을 함께 고려하여 대화 중 발화자의 감정을 인식하는 멀티모달 모델을 제안한다. 한국어 멀티모달 감정 데이터셋으로 KEMDy19를 활용하여 감정반응 적용유무에 대한 단일모달 모델을 구현하고 이를 기반으로 멀티모달 모델을 개발하였다. 그 결과 텍스트 기반의 단일모달 모델은 감정반응을 적용한 모델이 적용하지 않은 모델보다 약 10.1% 높은 성능을 보였고, 멀티모달 모델의 경우에는 8.1% 가량 높은 성능을 보였다. 이는 대화에서 감정반응을 고려하는 것이 감정인식 기술에 중요할 수 있다는 점을 시사하며 본 연구를 토대로 멀티모달 모델을 이용하여 더욱 향상된 공감형 AI 에이전트가 개발되기를 기대한다.

## 감정 반응과 감정 표현
![image](https://user-images.githubusercontent.com/130694680/233250368-01cb734c-875b-4dbe-b26c-475b2d70f204.png)

- 감정 반응(Emotion Response, {})

  두 사람의 대화에서 상대방의 발화를 듣고 내부적으로 생성되는 감정적 반응

- 감정 표현(Emotion Expression, [])

  발화자가 발화할 때 언어적 표현, 비언어적 행동을 통해 감정을 외부적으로 표현

## 모델 아키텍처
![image](https://user-images.githubusercontent.com/130694680/233250375-4c770470-3332-46c2-a68c-1487f5a9fe0e.png)

### Data shape
- Text

```
X_train (8721, 9) y_train (8721, )

X_test (2230, 9) y_test (2230, )
```
하나의 발화 세그먼트에서 2개 이상의 감정 레이블이 존재하는 경우, 감정 레이블 개수만큼 발화 세그먼트를 중복 생성하여 발화 세그먼트 당 단일 감정 레이블로 변환하였다. 감정 레이블은 있지만 텍스트가 없는 데이터 4개와 텍스트는 있지만 감정 레이블이 없는 데이터 1개를 삭제하여 총 10,951개의 데이터셋을 구성하였다. 감정반응 반영을 위해 감정반응과 텍스트를 결합시켜 감정반응을 포함한 텍스트 데이터셋을 생성하였다. 이후, KoBERT 모델의 토큰화를 통해 전처리한 데이터를 KoBERT 모델에 입력하여 감정을 분류하였다.
- EDA

```
X_train (5968, 172) y_train (5968, )

X_test (2021, 172) y_test (2021, )
```
EDA 데이터는 발화 세그먼트 별로 shape이 동일하지 않아 zero padding을 이용해 가장 길게 측정된 EDA 데이터의 shape으로 맞추었다. 이로 인해 실제 측정된 데이터가 아닌 0으로 입력된 데이터가 존재하게 되어 분석의 정확도를 향상시키기 위해 0으로 입력된 데이터의 개수를 줄이고자 하였다. 발화 세그먼트 당 가장 길게 측정된 EDA의 측정 횟수(172번)의 절반 이상인 데이터가 약 1.68%만 존재함에 따라 측정 횟수가 86번 이상인 데이터를 제거하여 총 7,989개의 데이터를 활용하였다. 또한, 감정반응 반영을 위해 발화자의 발화 이전 EDA 값을 학습 데이터에 연결하여 데이터셋을 생성하고, 최적 파라미터를 적용한 XGBClassifier 모델에 입력하여 감정을 분류하였다.
- Audio

```
X_train (9360, 128, 3300) y_train (9360, 7)

X_test (2095, 128, 3300) y_test (2095, 7)
```
![image](https://user-images.githubusercontent.com/130694680/233250400-7cf10de9-c6ba-4b57-9f1c-cde3ff4c6f2d.png)

Train 음성 데이터는 14초 이하의 길이를 가진 데이터가 95%로 구성되어 데이터 불균형을 보완하기 위해 14초를 초과하는 음성 데이터에 대해 음성 길이의 평균값인 10초를 기준으로 5초 간격의 sliding window를 적용하여 총 1,665개의 동일한 발화 세그먼트의 음성 데이터를 생성하였다. Test 데이터에 대해서도 Train 데이터와 동일한 방법을 적용하여 전체 음성 길이 분포에서 상위 5%에 해당하는 데이터를 데이터 평균 길이인 10초의 음성 파일로 변환하였다. 음성신호는 데이터 특성상 감정표현 정보만 담고 있으므로 감정반응을 적용한 데이터셋은 구성하지 않았다. 음성 데이터의 특징 값을 추출하기 위해 MFCC(Mel-Frequency Cepstral Coefficient)를 적용하고 데이터의 최대 길이를 기반으로 패딩 작업을 수행한 후, CNN을 활용하여 감정을 분류하였다(그림 3).

## 모델 성능 평가
![image](https://user-images.githubusercontent.com/130694680/231962041-547ca899-2c50-4076-830b-76a94f454bca.png)

## 실험 결과
![image](https://user-images.githubusercontent.com/130694680/231944849-ae8b71ed-e63b-4ce7-9541-3c38dc31247e.png)
