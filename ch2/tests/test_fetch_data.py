import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

from ch2 import fetch_data as fd
import numpy as np
import pandas as pd

DATA_PATH = '../housing/housing.csv'


def test_load_housing_data():
    data = fd.load_housing_data('../housing/housing.csv')
    print(data.head())
    assert data.shape == (20640, 10)


def test_permutation():
    # arr=np.random.permutation(10)
    # assert len(arr)==10
    data = fd.load_housing_data('../housing/housing.csv')
    train, test = fd.split_train_test(data, 0.2)
    assert len(train) == 16512
    assert len(test) == 4128


def test_train_test_by_id():
    data = fd.load_housing_data('../housing/housing.csv')
    ids = data.reset_index()
    train, test = fd.split_train_test_by_id(ids, 0.2, 'index')
    assert len(train) == 16512
    assert len(test) == 4128


def test_train_test_by_sklearn():
    from sklearn.model_selection import train_test_split
    data = fd.load_housing_data('../housing/housing.csv')
    ids = data.reset_index()
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    assert len(train) == 16512
    assert len(test) == 4128


def test_income_cut():
    from sklearn.model_selection import train_test_split
    data = fd.load_housing_data('../housing/housing.csv')
    fd.income_cut(data)
    # ids = data.reset_index()
    # train, test = train_test_split(data, test_size=0.2, random_state=42)
    # assert len(train) == 16512
    # assert len(test) == 4128


def test_set_check():
    crc = [fd.set_Check(i, 0.2) for i in range(2 ** 32)]
    crc_pd = pd.DataFrame(crc)
    assert crc == False


def test_visual_data():
    data = fd.load_housing_data('../housing/housing.csv')
    fd.visual_data(data)

    plt.show()


def test_visual_dataildata():
    data = fd.load_housing_data('../housing/housing.csv')
    data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.3, s=data['population'] / 100, label="population",
              figsize=(10, 7),
              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.show()


def test_corrmatrix():
    data = fd.load_housing_data(DATA_PATH)
    corr_matrix = data.corr()
    corr = corr_matrix['median_house_value'].sort_values(ascending=True)
    print(corr)


def test_scatter_mateix():
    data = fd.load_housing_data(DATA_PATH)
    corr_matrix = data.corr()
    from pandas.plotting import scatter_matrix
    attrs = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    scatter_matrix(data[attrs], figsize=(12, 8))
    data.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.2)
    plt.show()


def test_combine_scatter_mateix():
    data = fd.load_housing_data(DATA_PATH)
    data['rooms_pre_household'] = data['total_rooms'] / data['households']
    data['bedrooms_pre_room'] = data['total_bedrooms'] / data['total_rooms']
    data['population_per_household'] = data['population'] / data['households']
    corr_matrix = data.corr()
    corr = corr_matrix['median_house_value'].sort_values(ascending=False)
    print(corr)
    # from pandas.plotting import scatter_matrix
    # attrs = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    # scatter_matrix(data[attrs], figsize=(12, 8))
    # data.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.2)
    # plt.show()


def test_clear_data():
    data = fd.load_housing_data(DATA_PATH)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    num = data.drop('ocean_proximity', axis=1)
    imputer.fit(num)
    X = imputer.transform(num)
    pd_data = pd.DataFrame(X, columns=num.columns, index=num.index)
    assert np.allclose(data.median().values, imputer.statistics_, 0.0001)
    print(imputer.statistics_)


def test_encoder():
    from sklearn.preprocessing import OrdinalEncoder
    data = fd.load_housing_data(DATA_PATH)
    ordinal_encoder = OrdinalEncoder()
    cat = data[['ocean_proximity']]
    cat_encoded = ordinal_encoder.fit_transform(cat)
    print(cat_encoded[:10])
    print(ordinal_encoder.categories_)


def test_oneHotEncoder():
    from sklearn.preprocessing import OneHotEncoder
    cat_encoder = OneHotEncoder()
    data = fd.load_housing_data(DATA_PATH)
    cat_1hot = cat_encoder.fit_transform(data)
    print(cat_1hot.toarray())


from ch2.combine_attributes_addr import CombinedAttributesAdder


def test_transform():
    data = fd.load_housing_data(DATA_PATH)
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    data_extra_attribs = attr_adder.transform(data.values)
    print(data_extra_attribs)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_num_data():
    data = fd.load_housing_data(DATA_PATH)
    return data.drop(['ocean_proximity'],axis=1)


def test_pipeline():
    data = fd.load_housing_data(DATA_PATH)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    num_data = data.drop(['ocean_proximity'],axis=1)
    housing_num_tr = num_pipeline.fit_transform(num_data)
    print(num_data.shape)
    print(housing_num_tr.shape)

def create_pipe_data():
    data = fd.load_housing_data(DATA_PATH)
    num_data= data.drop(['ocean_proximity'], axis=1)

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    num_attribs=list(num_data)
    cat_attribs=['ocean_proximity']
    full_pipeline=ColumnTransformer([
        ('num',num_pipeline,num_attribs),
        ('cat',OneHotEncoder(),cat_attribs)
    ])
    data_prepared=full_pipeline.fit_transform(data)
    return data_prepared

def test_column_pipeline():
    data_prepared=create_pipe_data()
    print(data_prepared.shape)

def test_linear_reg():
    data=create_pipe_data()
    from sklearn.linear_model import LinearRegression
    lin_reg=LinearRegression()
    lin_reg.fit(data,data['median_house_value'])