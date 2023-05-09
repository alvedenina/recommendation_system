import pandas as pd
from server.database import engine

CHUNKSIZE = 200000


def batch_load_sql(query: str) -> pd.DataFrame:
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features(table):
    df = batch_load_sql(f"SELECT * FROM {table} LIMIT 10000000")
    return df


def one_hot_encoding(data: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    categorical_data_part = pd.get_dummies(data[cat_cols],
                                           prefix=cat_cols,
                                           drop_first=True
                                           )
    data = data.drop(cat_cols, axis=1)
    data = pd.concat((data, categorical_data_part), axis=1)
    return data
