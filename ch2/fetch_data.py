import os
import pandas as pd
from pandas import DataFrame
import numpy as np
from zlib import crc32

HOUSING_PATH = os.path.join('./housing/housing.csv')


def load_housing_data(path=HOUSING_PATH) -> DataFrame:
    return pd.read_csv(path)


def show_hist(data: DataFrame):
    data.hist(bins=100, figsize=(20, 15))
    import matplotlib.pyplot as plt
    plt.show()


def split_train_test(data: DataFrame, test_ratio: float):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indexs = shuffled_indices[0:test_set_size]
    train_indexs = shuffled_indices[test_set_size:]
    return data.iloc[train_indexs], data.iloc[test_indexs]


def split_train_test_by_id(data: DataFrame, test_ratio: float, id_column: str):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: set_Check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


def set_Check(identifier, test_radio: float):
    return crc32(np.int64(identifier)) & 0xffffffff < test_radio * 2 ** 32

def income_cut(data:DataFrame):
    data["income_cut"]=pd.cut(data['median_income'],bins=[0.,1.5,3.0,4.5,6,np.inf],labels=[1,2,3,4,5])
    from sklearn.model_selection import StratifiedShuffleSplit
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    spliits=split.split(data, data['income_cut'])
    for train_index,test_index in spliits:
        strat_train_set=data.loc[train_index]
        strat_test_set=data.loc[test_index]
        print(strat_train_set['income_cut'].value_counts())

def visual_data(data:DataFrame,alpha=0.5):
    data.plot(kind="scatter",x="longitude",y="latitude",alpha=alpha)



if __name__ == '__main__':
    data = load_housing_data()
    print(data.info())
    print(data.columns)
    # pd.set_option('')
    for col in data.columns.values:
        print(data[col].value_counts())
    show_hist(data)
# housing = load_housing_data()
# pd.set_option('display.max_columns', 500)
# print(housing.head())
