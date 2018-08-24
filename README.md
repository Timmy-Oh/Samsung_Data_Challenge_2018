# Samsung_Data_Challenge_2018
Samsung_Data_Challenge_2018


다음을 쥬피터를 이용하여 실행 : 학습 코드 main_train.ipynb, 테스트 코드 main._test.ipynb

실행 코드는 학습코드(main_train)와 테스트 코드(main_test)로 나누어진다.
쥬피터 노트북을 실행시켜 main_train.ipynb를 키고 셀을 실행시키면 학습이 시작된다. 
학습 데이터는 "./data/dataset_kor/교통사망사고정보/Kor_Train_교통사망사고정보(12.1~17.6).csv" 에 존재한다
학습 코드는 총 3개의 모델을 학습하며 학습된 웨이트는 ckpt폴더 아래 모델별 Best 모델만 저장되며 모든 학습이 끝나면 "Train is Ended, Do Test" 라는 메시지가 나타난다. 학습은 약 5분정도 걸린다 (GPU 기준)
학습이 완료되면 main_test.ipynb를 키고 셀을 실행시키면 학습된 모델들을 이용해 테스트 데이터에 대해 예측을 한다.
테스트 데이터는 "./data/test_kor.csv"에, 결과 데이터는 "./data/result_kor.csv"에 존재한다.
위의 2개의 데이터를 이용해 나온 최종 결과 데이터는 "./test_export/" 폴더 아래에 저장된다.
만약을 위해 학습코드와 테스트코드는 py로도 저장해 두었다. 해당경로 CMD에서 다음을 실행하면된다. (커맨드실행은 가끔 오류가 생기므로 노트북을 사용하자)
 $ python main_train.py
 $ python main_test.py
