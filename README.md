# 2023 제2회 ETRI 휴먼이해 인공지능 논문경진대회
본 대회는 한국전자통신연구원(ETRI)이 주최하고 과학기술정보통신부와 국가과학기술연구회(NST)가 후원합니다.


- 멀티모달 감정 데이터셋 활용 감정인식 기술 분야 [바로가기](https://aifactory.space/competition/detail/2234)
- KEMDy19 (성우 대상 상황극) 데이터셋 [바로가기](https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR)


# 논문 주제
## Abstract
 최근 인공지능 기술의 발전으로 대화에서의 감정인식 기술에 대한 연구가 지속적으로 이루어지고 있다. 기존 연구에서는 단일 모달보다 멀티 모달 모델을 사용하여 감정을 분류하는 것에 초점을 맞추고 있지만, 더 나아가 대화에서의 문맥이나 감정 반응을 고려해 발화자의 감정을 인식하는 방법에 대해 제안하고자 한다. ‘KEMDy19’ 데이터셋을 사용하여 감정을 분류한 결과 감정반응을 적용한 모델의 성능이 적용하지 않은 모델 성능보다 약 0.2%-10.1% 정도 향상되었고, 단일 모달을 사용했을 때 보다 멀티 모달 모델을 기반으로 감정을 분류했을 때 향상된 성능을 보였다.

## 감정 반응과 감정 표현
![image](https://user-images.githubusercontent.com/130694680/231942714-d344cdcf-2529-46e1-8823-e6265f6d7192.png)

- 감정 반응(Emotion Response, {})

  두 사람의 대화에서 상대방의 발화를 듣고 내부적으로 생성되는 감정적 반응

- 감정 표현(Emotion Expression, [])

  발화자가 발화할 때 언어적 표현, 비언어적 행동을 통해 감정을 외부적으로 표현

## 모델 아키텍처
![image](https://user-images.githubusercontent.com/130694680/231943514-e450f8d5-8db9-420a-ba6f-e5999c6e18cc.png)

### Data shape
- Text

X_train (8721, 9) y_train (8721, )

X_test (2230, 9) y_test (2230, )

- EDA

X_train (5968, 172) y_train (5968, )

X_test (2021, 172) y_test (2021, )

- Audio

X_train (9360, 128, 3300) y_train (9360, 7)

X_test (2095, 128, 3300) y_test (2095, 7)


## 모델 성능 평가
![image](https://user-images.githubusercontent.com/130694680/231944234-90344d42-c088-4543-86c6-521405c92905.png)

## 실험 결과
![image](https://user-images.githubusercontent.com/130694680/231944849-ae8b71ed-e63b-4ce7-9541-3c38dc31247e.png)
