import pandas as pd

#--------- 파일 불러오기 --------------------------------#

df_train = pd.read_csv('datasets/train.csv')
df_test = pd.read_csv('datasets/test.csv')

#--------- 데이터셋 레이블(속성, attribute) 보기 ---------#
df_train.keys()
    # Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], dtype='object')

#--------- 데이터셋 속성 리스트로 반환 -------------------#

ls_df_train = list(df_train.keys())
ls_df_test = list(df_test.keys())

#--------- 훈련 데이터셋 속성 리스트를 키값으로 반환 ------#

num_ls = [0, 1, 2, 5, 6, 7, 9]
num_ls_df_train = []
for i in num_ls:
    num_ls_df_train.append(ls_df_train[i])

#--------- 테스트 데이터셋 속성 리스트를 키값으로 반환 -----#

num_ls_2 = [0, 1, 4, 5, 6, 8]
num_ls_df_test = []
for i in num_ls_2:
    num_ls_df_test.append(ls_df_test[i])

#--------- 숫자 속성만을 갖는 데이터프레임 생성 ------------#

filt_df_train = df_train.loc[:,num_ls_df_train]
filt_df_test = df_test.loc[:,num_ls_df_test]

filt_df_train.info()
    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 891 entries, 0 to 890
    # Data columns (total 7 columns):
    # PassengerId    891 non-null int64
    # Survived       891 non-null int64
    # Pclass         891 non-null int64
    # Age            714 non-null float64
    # SibSp          891 non-null int64
    # Parch          891 non-null int64
    # Fare           891 non-null float64
    # dtypes: float64(2), int64(5)
    # memory usage: 48.9 KB

    # 177 loss in Age attribute = 891 - 714

#---------- PassengerId 객수 확인 -----------------------#
len(filt_df_train['PassengerId'].unique())

filt_df_test.info()
    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 418 entries, 0 to 417
    # Data columns (total 6 columns):
    # PassengerId    418 non-null int64
    # Pclass         418 non-null int64
    # Age            332 non-null float64
    # SibSp          418 non-null int64
    # Parch          418 non-null int64
    # Fare           417 non-null float64
    # dtypes: float64(2), int64(4)
    # memory usage: 19.7 KB

    # 86 loss in Age attribute = 418 - 332
    # 1 loss in Fare attribute = 418 - 417
#---------- PassengerId 객수 확인 -----------------------#
len(filt_df_test['PassengerId'].unique())

# NaN 처리할 것.
