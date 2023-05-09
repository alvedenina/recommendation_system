import pandas as pd
from catboost import CatBoostClassifier
import datetime

begin_time = datetime.datetime.now()

df = pd.read_csv('data/data_for_learning.csv', index_col=0)
df_learn = df.drop(['user_id', 'post_id'], axis=1).drop_duplicates()
X_train = df_learn.drop('target', axis=1)
y_train = df_learn['target']
cat_cols = ['hour', 'day_of_week', 'month', 'city', 'exp_group', 'TextCluster']

catboost = CatBoostClassifier(verbose=0)
catboost.fit(X_train,
             y_train,
             cat_features=cat_cols)
catboost.save_model('model_vedenina',
                    format="cbm")

print(f"Время работы алгоритма: {datetime.datetime.now() - begin_time}")
