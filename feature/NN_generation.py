import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import entropy
from itertools import product
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition


def date_proc(x):
    m = int(x[4:6])
    if m == 0:
        m = 1
    return x[:4] + '-' + str(m) + '-' + x[6:]


def date_tran(df, fea_col):
    for f in tqdm(fea_col):
        df[f] = pd.to_datetime(df[f].astype('str').apply(date_proc))
        df[f + '_year'] = df[f].dt.year
        df[f + '_month'] = df[f].dt.month
        df[f + '_day'] = df[f].dt.day
        df[f + '_dayofweek'] = df[f].dt.dayofweek
    return (df)


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def count_coding(df, fea_col):
    for f in fea_col:
        df[f + '_count'] = df[f].map(df[f].value_counts())
    return (df)


def cut_group(df, cols, num_bins=50):
    for col in cols:
        all_range = int(df[col].max() - df[col].min())
        bin = [i * all_range / num_bins for i in range(all_range)]
        df[col + '_bin'] = pd.cut(df[col], bin, labels=False)
    return df


def cross_cat_num(df, num_col, cat_col):
    for f1 in tqdm(cat_col):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_col):
            feat = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max', '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
            })
            df = df.merge(feat, on=f1, how='left')
    return (df)


### 类别特征的二阶交叉
def cross_qua_cat_num(df):
    for f_pair in tqdm([
        ['model', 'brand'], ['model', 'regionCode'], ['brand', 'regionCode']
    ]):
        ### 共现次数
        df['_'.join(f_pair) + '_count'] = df.groupby(f_pair)['SaleID'].transform('count')
        ### n unique、熵
        df = df.merge(df.groupby(f_pair[0], as_index=False)[f_pair[1]].agg({
            '{}_{}_nunique'.format(f_pair[0], f_pair[1]): 'nunique',
            '{}_{}_ent'.format(f_pair[0], f_pair[1]): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=f_pair[0], how='left')
        df = df.merge(df.groupby(f_pair[1], as_index=False)[f_pair[0]].agg({
            '{}_{}_nunique'.format(f_pair[1], f_pair[0]): 'nunique',
            '{}_{}_ent'.format(f_pair[1], f_pair[0]): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=f_pair[1], how='left')
        ### 比例偏好
        df['{}_in_{}_prop'.format(f_pair[0], f_pair[1])] = df['_'.join(f_pair) + '_count'] / df[f_pair[1] + '_count']
        df['{}_in_{}_prop'.format(f_pair[1], f_pair[0])] = df['_'.join(f_pair) + '_count'] / df[f_pair[0] + '_count']
    return (df)


class MeanEncoder:
    def __init__(self, categorical_features, n_splits=10, target_type='classification', prior_weight_func=None):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode

        :param n_splits: the number of splits used in mean encoding

        :param target_type: str, 'regression' or 'classification'

        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        """

        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()

        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()

        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg([('mean', 'mean'), ('beta', 'size')])
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, X, y):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target,
                        self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None,
                        self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new

    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()

        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits

        return X_new


if __name__ == '__main__':

    path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
    input_path = path + '/data/'
    train_path = input_path + 'used_car_train_20200313.csv'
    test_path = input_path + 'used_car_testB_20200421.csv'
    ## 通过Pandas对于数据进行读取 (pandas是一个很友好的数据读取函数库)
    Train_data = reduce_mem_usage(pd.read_csv(train_path, sep=' '))
    TestA_data = reduce_mem_usage(pd.read_csv(test_path, sep=' '))

    print('Train data shape:', Train_data.shape)
    print('Test data shape:', TestA_data.shape)

    concat_data = pd.concat([Train_data, TestA_data], axis=0, ignore_index=True)

    # 1.各个空值的处理
    concat_data['notRepairedDamage'] = concat_data['notRepairedDamage'].replace('-', 0).astype('float16')
    concat_data = concat_data.fillna(concat_data.mode().iloc[0, :])

    # 2.异常值的处理
    concat_data['power'][concat_data['power'] > 600] = 600
    concat_data['power'][concat_data['power'] < 1] = 1

    concat_data['v_13'][concat_data['v_13'] > 6] = 6
    concat_data['v_14'][concat_data['v_14'] > 4] = 4

    # concat_data.drop(concat_data[concat_data['seller'] == 1].index, inplace=True)

    # 3.特征之间进行了相互组合
    for i in ['v_' + str(i) for i in range(14)]:
        for j in ['v_' + str(i) for i in range(14)]:
            concat_data[str(i) + '+' + str(j)] = concat_data[str(i)] + concat_data[str(j)]
    for i in ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'power', 'kilometer', 'notRepairedDamage',
              'regionCode']:
        for j in ['v_' + str(i) for i in range(14)]:
            concat_data[str(i) + '*' + str(j)] = concat_data[i] * concat_data[j]

    # 4.日期的处理
    # # 提取日期信息
    date_cols = ['regDate', 'creatDate']
    concat_data = date_tran(concat_data, date_cols)

    data = concat_data.copy()
    data['used_time1'] = (pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce') -
                          pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')).dt.days
    data['used_time1_year'] = round(data['used_time1'] / 365, 1)
    data['used_time2'] = (pd.datetime.now() - pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')).dt.days
    data['used_time3'] = (
            pd.datetime.now() - pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce')).dt.days

    # 5.count编码 与 分桶操作
    count_list = ['regDate', 'creatDate', 'model', 'brand', 'regionCode', 'bodyType', 'fuelType', 'name',
                  'regDate_year',
                  'regDate_month', 'regDate_day', 'notRepairedDamage', 'gearbox',
                  'regDate_dayofweek', 'creatDate_month', 'creatDate_day', 'creatDate_dayofweek', 'kilometer']

    data = count_coding(data, count_list)
    cut_cols = ['power'] + ['used_time1', 'used_time2', 'used_time3']
    data = cut_group(data, cut_cols, 50)

    # 6.用数值特征对类别特征做统计刻画   与   类别与类别特征的统计刻画
    cross_cat = ['model', 'brand', 'regDate_year']
    cross_num = ['v_0', 'v_3', 'v_4', 'v_8', 'v_12', 'power']
    data = cross_cat_num(data, cross_num, cross_cat)  # 一阶交叉

    data = cross_qua_cat_num(data)

    # 7.选择特征列
    numerical_cols = data.columns
    # print(numerical_cols)

    cat_fea = ['SaleID', 'offerType', 'seller']
    feature_cols = [col for col in numerical_cols if col not in cat_fea]
    feature_cols = [col for col in feature_cols if col not in ['price']]

    ## 提前特征列，标签列构造训练样本和测试样本
    X_data = data.iloc[:len(Train_data), :][feature_cols]
    Y_data = Train_data['price']
    X_test = data.iloc[len(Train_data):, :][feature_cols]

    print(X_data.shape)

    # 8.对某些类别特征进行mean_encoding类别编码
    class_list = ['model', 'brand', 'name', 'regionCode', 'bodyType', 'fuelType', 'regDate_year',
                  'creatDate_year'] + date_cols
    MeanEnocodeFeature = class_list  # 声明需要平均数编码的特征
    ME = MeanEncoder(MeanEnocodeFeature, target_type='regression')  # 声明平均数编码的类
    X_data = ME.fit_transform(X_data, Y_data)  # 对训练数据集的X和y进行拟合
    # x_train_fav = ME.fit_transform(x_train,y_train_fav)#对训练数据集的X和y进行拟合
    X_test = ME.transform(X_test)  # 对测试集进行编码

    X_data['price'] = Train_data['price']
    print('mean_encoding之后：', X_data.shape)

    # 9.再对某些类别特征做target_encoding，不仅可以做均值编码，还可以做标准差编码、中位数编码等
    enc_cols = []
    stats_default_dict = {
        'max': X_data['price'].max(),
        'min': X_data['price'].min(),
        'median': X_data['price'].median(),
        'mean': X_data['price'].mean(),
        'sum': X_data['price'].sum(),
        'std': X_data['price'].std(),
        'skew': X_data['price'].skew(),
        'kurt': X_data['price'].kurt(),
        'mad': X_data['price'].mad()
    }
    ### 暂且选择这三种编码
    enc_stats = ['max', 'min', 'mean']
    skf = KFold(n_splits=10, shuffle=True, random_state=42)
    for f in tqdm(
            ['regionCode', 'brand', 'regDate_year', 'creatDate_year', 'kilometer', 'model', 'bodyType', 'fuelType']):
        enc_dict = {}
        for stat in enc_stats:
            enc_dict['{}_target_{}'.format(f, stat)] = stat
            X_data['{}_target_{}'.format(f, stat)] = 0
            X_test['{}_target_{}'.format(f, stat)] = 0
            enc_cols.append('{}_target_{}'.format(f, stat))
        for i, (trn_idx, val_idx) in enumerate(skf.split(X_data, Y_data)):
            trn_x, val_x = X_data.iloc[trn_idx].reset_index(drop=True), X_data.iloc[val_idx].reset_index(drop=True)
            enc_df = trn_x.groupby(f, as_index=False)['price'].agg(enc_dict)
            val_x = val_x[[f]].merge(enc_df, on=f, how='left')
            test_x = X_test[[f]].merge(enc_df, on=f, how='left')
            for stat in enc_stats:
                val_x['{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].fillna(
                    stats_default_dict[stat])
                test_x['{}_target_{}'.format(f, stat)] = test_x['{}_target_{}'.format(f, stat)].fillna(
                    stats_default_dict[stat])
                X_data.loc[val_idx, '{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].values
                X_test['{}_target_{}'.format(f, stat)] += test_x['{}_target_{}'.format(f, stat)].values / skf.n_splits
    print('target_encoding之后：', X_data.shape)

    # 10.删除某些列
    drop_list = ['regDate', 'creatDate', 'brand_power_min', 'regDate_year_power_min']
    x_train = X_data.drop(drop_list + ['price'], axis=1)
    x_test = X_test.drop(drop_list, axis=1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    print(x_train.shape)

    # 11.所有特征归一化
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(pd.concat([x_train, x_test]).values)
    all_data = min_max_scaler.transform(pd.concat([x_train, x_test]).values)

    # 12.利用pca进行特征降维
    data_dim = 150
    pca = decomposition.PCA(n_components=data_dim)
    all_pca = pca.fit_transform(all_data)

    X_pca = all_data[:len(x_train)]
    test = all_data[len(x_train):]
    y = Train_data['price'].values

    print('数据降维后的维度：', X_pca.shape)

    # 13.将特征保存下来
    output_path = path + '/user_data/'
    np.save(output_path + 'train_x_nn.npy', X_pca)
    np.save(output_path + 'train_y_nn.npy', y)
    np.save(output_path + 'test_x_nn.npy', test)

    print('NN模型数据已经准备完毕~~~~~~~')
