import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

df_train = pd.read_csv('C:/Users/User/Desktop/面試題目/data-question 2/train.csv')#訓練資料載入
df_test = pd.read_csv('C:/Users/User/Desktop/面試題目/data-question 2/test.csv')

#----------------------------------------------------------------------------
num1 = len(df_train)
df_train.dropna()#drop掉空值欄位
num2 = len(df_train)
if(num1 == num2):#有drop掉欄位,前後欄位數會不一樣
    print("此資料無缺失值")
else:
    print("資料存在缺失值")#如有缺失值,使用fillna填補該欄位平均值
    #df[''] = df[''].fillna(df[''].mean())

#----------------------------------------------------------------------------
X = df_train.iloc[:,1:18]
X = X.drop(columns =["SpecialDay"])
y = df_train.iloc[:,18:]

#----------------------------------------------------------------------------
def normalize(df): #正規化
    norm = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return norm
hotel = normalize(X)

#----------------------------------------------------------------------------
x_train,x_test ,y_train,y_test = train_test_split(hotel, y,test_size=0.2,random_state=42,stratify=y)#模型訓練資料切分
'''xgb_n_estimators = [int(x) for x in np.linspace(10, 500, 10)]# Number of trees to be used
xgb_max_depth = [int(x) for x in np.linspace(2, 20, 10)]# Maximum number of levels in tree
xgb_min_child_weight = [int(x) for x in np.linspace(1, 10, 10)]# Minimum number of instaces needed in each node
xgb_eta = [x for x in np.linspace(0.1, 0.6, 6)]# Learning rate
xgb_gamma = [int(x) for x in np.linspace(0, 0.5, 6)]# Minimum loss reduction required to make further partition
# Create the grid
xgb_grid = {'n_estimators': xgb_n_estimators,
            'max_depth': xgb_max_depth,
            'min_child_weight': xgb_min_child_weight,
            'eta': xgb_eta,
            'gamma': xgb_gamma}

xgb_base = XGBClassifier()
xgb_random = RandomizedSearchCV(estimator = xgb_base, param_distributions = xgb_grid,
                                n_iter = 200, cv = 3, verbose = 2,
                                random_state = 42, n_jobs = -1)

xgb_random.fit(x_train, y_train)# Fit the random search model
xgb_random.best_params_# Get the optimal parameters'''

#----------------------------------------------------------------------------
xgbrModel = XGBClassifier(n_estimators=60,min_child_weight=7,max_depth=6,gamma=0,learning_rate=0.1)#7
xgbrModel.fit(x_train,y_train)
y_pred = xgbrModel.predict(x_test)
print('訓練集:',xgbrModel.score(x_train,y_train))#0.9358024691358025
print('測試集:',xgbrModel.score(x_test,y_test))#0.9111111111111111

#----------------------------------------------------------------------------
plt.rcParams['font.sans-serif']=['Taipei Sans TC Beta']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True
plot_importance(xgbrModel,title='特徵重要性排序',xlabel='分數',ylabel='特徵')
plt.show()

#----------------------------------------------------------------------------
pickle.dump(xgbrModel,open("C:/Users/User/Desktop/面試題目/hotel.pickle.dat","wb"))
loaded_model = pickle.load(open("C:/Users/User/Desktop/面試題目/hotel.pickle.dat","rb"))
a_df = df_test.iloc[:,0:1]
df_test = df_test.iloc[:,1:18]
df_test = df_test.drop(columns =["SpecialDay"])
df_test = normalize(df_test)
pred = loaded_model.predict(df_test)
b_df = pd.DataFrame(pred)   
b_df.columns = ['HasRevenue']
new_df = pd.concat([a_df,b_df],axis=1)
new_df.to_csv("C:/Users/User/Desktop/面試題目/Answer.csv",encoding='utf_8_sig')

