import pandas as pd
import datetime
from server.database import engine
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from features import one_hot_encoding


def read_sql(df: pd.DataFrame) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {df}",
                       engine)


def user_data_processing(users: pd.DataFrame) -> pd.DataFrame:
    """
    OHE categorical features. Label encoding 'city'.
    """
    cat_col_users = ['country', 'os', 'source']
    users = one_hot_encoding(users, cat_col_users)
    le = preprocessing.LabelEncoder()
    le.fit(users['city'])
    users['city'] = le.transform(users['city'])
    return users


def post_data_processing(posts: pd.DataFrame) -> pd.DataFrame:
    """
    OHE categorical features. Text replaced by its length.
    """
    cat_col_post = ['topic']
    posts['text'] = round(posts['text'].str.len(), 2)
    posts = one_hot_encoding(posts, cat_col_post)
    return posts


def read_feed_data():
    return pd.read_sql("with a as ("
                       "select timestamp, user_id, post_id, action, target, "
                       "row_number() over (partition by user_id) as rn "
                       "from feed_data "
                       "order by user_id, rn) "
                       "select timestamp, user_id, post_id, action, target, rn "
                       "from a where rn <= 40 and action = 'view' limit 7000000",
                       engine)


def feed_data_processing(feeds: pd.DataFrame) -> pd.DataFrame:
    """
    Drop support features. Processing timestamp.
    """
    feeds = feeds.drop(['action', 'rn'], axis=1)
    feeds = feeds.sort_values('timestamp')
    feeds['timestamp'] = pd.to_datetime(feeds['timestamp'])
    feeds['hour'] = feeds['timestamp'].dt.hour
    feeds['day_of_week'] = feeds['timestamp'].dt.dayofweek
    feeds['month'] = feeds['timestamp'].dt.month
    feeds = feeds.drop(['timestamp'], axis=1)
    return feeds


def merging_and_dropping_duplicates(feed_data: pd.DataFrame,
                                    user_data: pd.DataFrame,
                                    post_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merging tables. Removal 'view' rows if it has 'like'.
    """
    df_merged = feed_data.merge(user_data, how='left', on='user_id') \
        .merge(post_data, how='left', on='post_id') \
        .drop_duplicates()
    data_group = df_merged.groupby(['user_id', 'post_id'])[['target']].mean() \
        .rename(columns={'target': 'mean_target'})
    df_merged = df_merged.merge(data_group[(data_group['mean_target'] > 0) & (data_group['mean_target'] < 1)],
                                how='left', on=['user_id', 'post_id'])
    df_merged['mean_target'] = df_merged['mean_target'].fillna(0)
    df_merged = df_merged[~((df_merged['target'] == 0) & (df_merged['mean_target'] > 0))] \
        .drop('mean_target', axis=1)
    return df_merged


def data_scaling(data: pd.DataFrame) -> pd.DataFrame:
    """
    Processing features with big value.
    """
    big_cols = ['age', 'mean_text']
    ct = ColumnTransformer([('StandardScaler', StandardScaler(), big_cols)])
    transformed_data = pd.DataFrame(ct.fit_transform(data)) \
        .rename(columns={0: 'age', 1: 'mean_text'}) \
        .set_index(data.index)
    data['age'] = transformed_data['age']
    data['mean_text'] = transformed_data['mean_text']
    return data


if __name__ == '__main__':
    begin_time = datetime.datetime.now()

    user_data = read_sql('user_data')
    user_data = user_data_processing(user_data)

    post_text_df = read_sql('post_text_df')
    post_text_df = post_data_processing(post_text_df)

    feed_data = read_feed_data()
    feed_data = feed_data_processing(feed_data)

    df = merging_and_dropping_duplicates(feed_data, user_data, post_text_df)

    # mean text length for each user
    df['mean_text'] = df['user_id'].map(df.groupby('user_id')['text'].mean())

    # mean count likes for each topic for each user
    topic_cols = ['topic_covid', 'topic_entertainment', 'topic_movie', 'topic_politics', 'topic_sport', 'topic_tech']
    for col in topic_cols:
        df[f'{col}_mean'] = round(df['user_id'].map(df.groupby('user_id')[col].mean()), 4)

    # data scaling
    df = data_scaling(df)

    df.to_csv('data/data_for_learning.csv')
    df = df.drop_duplicates(subset='user_id', keep='first')
    df.to_sql('vedenina_ai_features_lesson_22', con=engine, if_exists='replace', chunksize=10000)

    print(f"Время работы алгоритма: {datetime.datetime.now() - begin_time}")
