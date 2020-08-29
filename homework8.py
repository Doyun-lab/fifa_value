#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Import
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA

# 데이터 불러오기
datapath = './data/data.csv'
fifa = pd.read_csv(datapath)

# 행 18207개, 열 89개로 이루어져 있다
fifa.shape


# In[11]:


# 데이터 앞 부분 살펴보기
fifa.head()


# In[12]:


# 불필요한 열 제거 (이미지 링크 제거)
fifa = fifa.drop(["Unnamed: 0", "Photo", "Flag", "Club Logo"], axis = 1)
fifa


# In[13]:


# 데이터형 확인
fifa.info()
# 타겟변수가 object로 되어있어 regression을 위해 전처리 필요하다고 생각

# 결측치 확인
fifa.isnull().sum()


# In[14]:


# Value와 Wage 문자열 제거 
# 단위에 따라 숫자 곱해주기
def string_change(fifa_value):
    try:
        value = float(fifa_value[1:-1])
        unit = fifa_value[-1:]

        if unit == 'M':
            value = value * 1000000
        elif unit == 'K':
            value = value * 1000
    except ValueError:
        value = 0

    return value

fifa['Value'] = fifa['Value'].apply(string_change)
fifa['Wage'] = fifa['Wage'].apply(string_change)


# In[15]:


# Weight에 붙어있는 문자열 제거
for x in range(0, 18207):
    a = str(fifa['Weight'][x])
    weight = a[0:3]
    
    fifa['Weight'][x] = int(weight)
    
    print(weight)

    


# In[19]:


# 피파 랭킹 20위 안 국가들은 1, 아닌 국가들은 0
# 피파 랭킹이 높은 나라일수록 대체로 축구를 잘하므로 그 나라에 속한 선수들의 Value가 높을 것이라 생각
top20 = ['Belgium', 'France', 'Brazil', 'England', 'Uruguay', 
         'Croatia', 'Portugal', 'Spain', 'Argentina', 'Colombia', 
         'Mexico', 'Switzerland', 'Italy', 'Netherlands', 'Germany', 
         'Denmark', 'Chile', 'Sweden', 'Poland', 'Senegal']

def nation_top20(fifa):
    if (fifa.Nationality in top20):
        return 1
    else:
        return 0

fifa['top20'] = fifa.apply(nation_top20, axis = 1)

fifa['top20'].head(50)


# In[16]:


# 주발에 따라 Value가 차이날 수 있을 것이라 생각
def foot_change(fifa):
    if (fifa['Preferred Foot'] == 'Right'):
        return 1
    else:
        return 0

fifa['foot'] = fifa.apply(foot_change, axis = 1)

fifa['foot'].head(10)


# In[20]:


# 필요한 열만 불러오기
# NA 값이 있는 행 제거 
# 위에서 확인했듯이 NA값이 있는 행이 전체 데이터의 수보다 작은 수이므로 모델링에 영향을 크게 주지않을 것이라 판단
# 나이와 능력치, 세부 능력치 등을 주로 불러옴

fifa = fifa[['Age', 'Overall', 'Potential', 'Value', 'Wage', 'Special',
     'International Reputation', 'Weak Foot', 'Skill Moves', 'top20', 'foot',
     'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
      'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
      'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
      'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
      'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
      'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
      'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]

fifa = fifa.dropna(axis = 0)


fifa

print("#########################################################")


# In[22]:


# Linear Regression 1
# 세부능력치는 Overall(종합 능력치)에 포함되는 부분이라고 생각하여 제외하고,
# 나이, 능력치, 잠재력, 주급, 국제적 평판, 약발, 피파랭킹 20위 안 국가, 주발만 Feature로 선택하여 Modeling

features = ['Age', 'Overall', 'Potential', 'Wage', 
     'International Reputation', 'Weak Foot','top20', 'foot']

# k-fold 교차 검정 실시해줄 것
kf = KFold(n_splits = 10, shuffle = True)

mae = []
fold_idx = 1

for train_idx, test_idx in kf.split(fifa):
    print('fold {}'.format(fold_idx))
    train_d, test_d = fifa.iloc[train_idx], fifa.iloc[test_idx]
    
    train_y = train_d['Value']
    train_x = train_d[features]
    
    test_y = test_d['Value']
    test_x = test_d[features]
    
    model = LinearRegression()
    model.fit(train_x, train_y)
    
    # MAE 값 구하기
    score = make_scorer(mean_absolute_error)
    mean_mae = score(model, test_x, test_y)
 
    mae.append(mean_mae)
    
    fold_idx += 1

# Model 10개의 평균 MAE 값
print("----------------------------------------")
print("Linear Regression 1")
print(np.average(mae))
print("#########################################################")


# In[19]:


# Linear Regression 2
# 세부능력치도 모두 넣고 Modeling 하여 Linear Regression 1 과 비교

features = ['Age', 'Overall', 'Potential', 'Wage', 'Special',
     'International Reputation', 'Weak Foot', 'Skill Moves', 'top20',
     'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
      'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
      'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
      'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
      'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
      'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
      'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

# k-fold 교차 검정 실시해줄 것
kf = KFold(n_splits = 10, shuffle = True)

mae = []
fold_idx = 1

for train_idx, test_idx in kf.split(fifa):
    print('fold {}'.format(fold_idx))
    train_d, test_d = fifa.iloc[train_idx], fifa.iloc[test_idx]
    
    train_y = train_d['Value']
    train_x = train_d[features]
    
    test_y = test_d['Value']
    test_x = test_d[features]
    
    model = LinearRegression()
    model.fit(train_x, train_y)
    
    # MAE 값 구하기
    score = make_scorer(mean_absolute_error)
    mean_mae = score(model, test_x, test_y)
 
    mae.append(mean_mae)
    
    fold_idx += 1

# Model 10개의 평균 MAE 값
print("----------------------------------------")
print("Linear Regression 2")
print(np.average(mae))
print("#########################################################")

# Linear Regreesion 2 Model이 Linear Regression 1 Model 보다 좋지않은 결과
# Ridge, Lasso ... Modeling시에는 Linear Regression 1 Model의 Feature를 사용하여 Modeling 하기로 결정


# In[20]:


# Ridge

features = features = ['Age', 'Overall', 'Potential', 'Wage', 
     'International Reputation', 'Weak Foot','top20', 'foot']

# k-fold 교차 검정 실시해줄 것
kf = KFold(n_splits = 10, shuffle = True)

mae = []
fold_idx = 1

for train_idx, test_idx in kf.split(fifa):
    print('fold {}'.format(fold_idx))
    train_d, test_d = fifa.iloc[train_idx], fifa.iloc[test_idx]
    
    train_y = train_d['Value']
    train_x = train_d[features]
    
    test_y = test_d['Value']
    test_x = test_d[features]
    
    model = LinearRegression()
    model.fit(train_x, train_y)
    
    # MAE 값 구하기
    score = make_scorer(mean_absolute_error)
    mean_mae = score(model, test_x, test_y)
 
    mae.append(mean_mae)
    
    fold_idx += 1

# Model 10개의 평균 MAE 값
print("----------------------------------------")
print("Ridge")
print(np.average(mae))
print("#########################################################")


# In[22]:


# Lasso
    
features = ['Age', 'Overall', 'Potential', 'Wage', 
     'International Reputation', 'Weak Foot','top20', 'foot']

# k-fold 교차 검정 실시해줄 것
kf = KFold(n_splits = 10, shuffle = True)

mae = []
fold_idx = 1

for train_idx, test_idx in kf.split(fifa):
    print('fold {}'.format(fold_idx))
    train_d, test_d = fifa.iloc[train_idx], fifa.iloc[test_idx]
    
    train_y = train_d['Value']
    train_x = train_d[features]
    
    test_y = test_d['Value']
    test_x = test_d[features]
    
    model = LinearRegression()
    model.fit(train_x, train_y)
    
    # MAE 값 구하기
    score = make_scorer(mean_absolute_error)
    mean_mae = score(model, test_x, test_y)
 
    mae.append(mean_mae)
    
    fold_idx += 1

# Model 10개의 평균 MAE 값
print("----------------------------------------")
print("Lasso")
print(np.average(mae))
print("#########################################################")

# Linear Regression, Ridge Regression, Lasso Regression 모두 비슷한 성능을 보임


# In[23]:


# Random Forest Regressor (PCA)

features = ['Age', 'Overall', 'Potential', 'Wage', 
     'International Reputation', 'Weak Foot','top20', 'foot']

# k-fold 교차 검정 실시해줄 것
kf = KFold(n_splits = 10, shuffle = True)

mae = []
fold_idx = 1

for train_idx, test_idx in kf.split(fifa):
    print('fold {}'.format(fold_idx))
    train_d, test_d = fifa.iloc[train_idx], fifa.iloc[test_idx]
    
    train_y = train_d['Value']
    train_x = train_d[features]
    
    test_y = test_d['Value']
    test_x = test_d[features]
    
    model = RandomForestRegressor()
    model.fit(train_x, train_y)
    
    # MAE 값 구하기
    score = make_scorer(mean_absolute_error)
    mean_mae = score(model, test_x, test_y)
 
    mae.append(mean_mae)
    
    fold_idx += 1

# Model 10개의 평균 MAE 값
print("----------------------------------------")
print("Random Forest Regressor (PCA)")
print(np.average(mae))
print("#########################################################")


# In[ ]:


# Random Forest Regressor
    
features = ['Age', 'Overall', 'Potential', 'Wage', 
     'International Reputation', 'Weak Foot','top20', 'foot']

# k-fold 교차 검정 실시해줄 것
kf = KFold(n_splits = 10, shuffle = True)

mae = []
fold_idx = 1

for train_idx, test_idx in kf.split(fifa):
    print('fold {}'.format(fold_idx))
    train_d, test_d = fifa.iloc[train_idx], fifa.iloc[test_idx]
    
    train_y = train_d['Value']
    train_x = train_d[features]
    
    test_y = test_d['Value']
    test_x = test_d[features]
    
    model = RandomForestRegressor()
    model.fit(train_x, train_y)
    
    # MAE 값 구하기
    score = make_scorer(mean_absolute_error)
    mean_mae = score(model, test_x, test_y)
 
    mae.append(mean_mae)
    
    fold_idx += 1

# Model 10개의 평균 MAE 값
print("----------------------------------------")
print("최종 모델 : Random Forest Regressor")
print(np.average(mae))

# 최종 모델로 Random Forest를 선택했습니다
# Feature에 세부 능력치 (ex.Dribbling', 'Curve', 'FKAccuracy', 'LongPassing' ...) 등은
# Feature Overall에 모두 포함되는 수치이므로 제거해 보는 것도 괜찮을 것이라 생각하여 제거하고 Model을 만들었을 때
# 세부 능력치가 들어가지 않은 Model이 더 좋은 결과를 보여 위와 같은 Feautre를 선택했씁니다
# (선수 가치에는 선수의 실력뿐만 아닌 나이, 잠재력, 국가, 국제적 평판도 영향을 미칠 것이라 생각)

## Parameter Engineering ##
# n_estimators parameter(Tree 수)를 조정해면서 비교해보았으나 default 값이 제일 좋았습니다
# criterion = "mse" parameter를 mae로 변경하여 비교해보았으나 mse가 더 좋았습니다
# bootstrap sampling을 시행하지 않은 parameter로 변경하여 비교해봤을때 bootstrap 실행한 것이 더 좋았습니다
# PCA를 통해 transform 된 train_x, test_x 데이터를 사용해서 비교해봤을때 PCA를 사용하지 않은 것이 더 좋은 결과가 나왔습니다
# Feature를 늘려 세부 능력치도 모두 넣어 모델링을 했을 때보다 주요 Feature룰 통해 생성한 Model 결과가 더 좋았습니다.


# In[ ]:





# In[ ]:




