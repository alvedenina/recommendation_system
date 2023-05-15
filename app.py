import os
from catboost import CatBoostClassifier
from datetime import datetime
from fastapi import Depends, FastAPI, HTTPException
import pandas as pd
import hashlib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from server.schema import PostGet, Response
from data_processing import read_all_sql, post_data_processing

SALT = 'my_first_experiment'
app = FastAPI()


def get_model_path(path: str, type) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально
        MODEL_PATH = f'/workdir/user_input/model_{type}'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models(model_name):
    if model_name == 'model_test':
        model_path = get_model_path(model_name, 'test')
    else:
        model_path = get_model_path(model_name, 'control')
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def get_exp_group(user_id: int, salt, group_count) -> str:
    value_str = str(user_id) + salt
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    if value_num % group_count == 0:
        return 'control'
    else:
        return 'test'


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 5):
    user = user_data[user_data['user_id'] == id]
    user = user[['hour', 'day_of_week', 'month', 'gender', 'age', 'country', 'city',
                 'exp_group', 'os_iOS', 'source_organic', 'mean_text', 'topic_covid_mean',
                 'topic_entertainment_mean', 'topic_movie_mean', 'topic_politics_mean', 'topic_sport_mean',
                 'topic_tech_mean']]
    user['hour'] = time.hour
    user['day_of_week'] = time.weekday()
    user['month'] = time.month

    user_x_posts = pd.merge(user, posts_for_merge, how='cross')
    df_for_predictions = user_x_posts.drop(['post_id'], axis=1)
    df_for_predictions = df_for_predictions[cols_for_predictions]

    exp_group = get_exp_group(id, SALT, 2)
    if exp_group == 'control':
        df_for_predictions.drop(['TextCluster', 'DistanceTo1thCluster', 'DistanceTo2thCluster', 'DistanceTo3thCluster',
                                 'DistanceTo4thCluster', 'DistanceTo5thCluster', 'DistanceTo6thCluster',
                                 'DistanceTo7thCluster', 'DistanceTo8thCluster', 'DistanceTo9thCluster',
                                 'DistanceTo10thCluster', 'DistanceTo11thCluster', 'DistanceTo12thCluster',
                                 'DistanceTo13thCluster', 'DistanceTo14thCluster', 'DistanceTo15thCluster'], axis=1)
        X_for_pred = df_for_predictions.drop_duplicates(ignore_index=True)
        prediction = pd.DataFrame(model_control.predict_proba(X_for_pred).round(4), columns=['prob_0', 'prob_1'])
        # recommendations = model_control.apply(data)
    elif exp_group == 'test':
        X_for_pred = df_for_predictions.drop_duplicates(ignore_index=True)
        prediction = pd.DataFrame(model_test.predict_proba(X_for_pred).round(4), columns=['prob_0', 'prob_1'])
        # recommendations = model_test.apply(data)
    else:
        raise ValueError('unknown group')

    X_result = pd.concat((X_for_pred, prediction[['prob_1']]), axis=1)
    df_result = pd.merge(user_x_posts, X_result, how='left')
    best_posts_list = df_result.nlargest(limit, 'prob_1')['post_id'].tolist()
    recommended_posts = posts[posts['post_id'].isin(best_posts_list)]
    result_list = []
    for i in best_posts_list:
        post = recommended_posts[recommended_posts['post_id'] == i]
        p_id = post["post_id"].values[0]
        text = post['text'].values[0]
        topic = post['topic'].values[0]
        result_list.append(PostGet(**{"id": p_id, "text": text, "topic": topic}))
    return Response(**{"exp_group": exp_group, "recommendations": result_list})


def data_scaling_app(data: pd.DataFrame) -> pd.DataFrame:
    """
    Processing features with big value.
    """
    big_cols = ['text']
    ct = ColumnTransformer([('StandardScaler', StandardScaler(), big_cols)])
    transformed_data = pd.DataFrame(ct.fit_transform(data)) \
        .rename(columns={0: 'text'}) \
        .set_index(data.index)
    data['text'] = transformed_data['text']
    return data


model_test = load_models('model_test')
model_control = load_models('model_control')

user_data = read_all_sql("vedenina_ai_features_dl_10")
user_data = user_data.drop('index', axis=1)
cols_for_predictions = user_data.columns.drop(['user_id', 'post_id', 'target'])

posts = read_all_sql('vedenina_posts_dl_10')
posts_for_merge = posts.copy()
posts_for_merge = post_data_processing(posts_for_merge)

posts_for_merge = data_scaling_app(posts_for_merge)
