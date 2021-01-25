import os
import keras

import numpy as np
import pandas as pd
import keras.backend as K

from keras.callbacks import LearningRateScheduler
from keras.layers import Dense
from keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_train, y_train = self.data[0][0], self.data[0][1]
        y_pred3 = self.model.predict(X_train)
        y_pred = np.zeros((len(y_pred3),))
        y_true = np.zeros((len(y_pred3),))
        for i in range(len(y_pred3)):
            y_pred[i] = y_pred3[i]
        for i in range(len(y_pred3)):
            y_true[i] = y_train[i]
        trn_s = mean_absolute_error(y_true, y_pred)
        logs['trn_score'] = trn_s

        X_val, y_val = self.data[1][0], self.data[1][1]
        y_pred3 = self.model.predict(X_val)
        y_pred = np.zeros((len(y_pred3),))
        y_true = np.zeros((len(y_pred3),))
        for i in range(len(y_pred3)):
            y_pred[i] = y_pred3[i]
        for i in range(len(y_pred3)):
            y_true[i] = y_val[i]
        val_s = mean_absolute_error(y_true, y_pred)
        logs['val_score'] = val_s
        print('trn_score', trn_s, 'val_score', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)


def NN_model(input_dim):
    init = keras.initializers.glorot_uniform(seed=1)
    model = keras.models.Sequential()
    model.add(Dense(units=300, input_dim=input_dim, kernel_initializer=init, activation='softplus'))
    # model.add(Dropout(0.2))
    model.add(Dense(units=300, kernel_initializer=init, activation='softplus'))
    # model.add(Dropout(0.2))
    model.add(Dense(units=64, kernel_initializer=init, activation='softplus'))
    model.add(Dense(units=32, kernel_initializer=init, activation='softplus'))
    model.add(Dense(units=8, kernel_initializer=init, activation='softplus'))
    model.add(Dense(units=1))
    return model


def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 20 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.6)
        print("lr changed to {}".format(lr * 0.6))
    return K.get_value(model.optimizer.lr)


if __name__ == '__main__':

    path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
    nn_data_path = path + '/user_data/'
    X_pca = np.load(nn_data_path + 'train_x_nn.npy')
    y = np.load(nn_data_path + 'train_y_nn.npy')
    test = np.load(nn_data_path + 'test_x_nn.npy')
    test_path = '../data/used_car_testB_20200421.csv'
    train_path = '../data/used_car_train_20200313.csv'
    b_size = 2000
    max_epochs = 145
    oof_pred = np.zeros((len(X_pca),))
    reduce_lr = LearningRateScheduler(scheduler)
    n_splits = 6
    kf = KFold(n_splits=n_splits, shuffle=True)
    avg_mae = 0
    sub = pd.read_csv(test_path, sep=' ')[['SaleID']].copy()
    sub['price'] = 0

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_pca, y)):
        print('fold:', fold)
        X_train, y_train = X_pca[trn_idx], y[trn_idx]
        X_val, y_val = X_pca[val_idx], y[val_idx]

        model = NN_model(X_train.shape[1])
        simple_adam = keras.optimizers.Adam(lr=0.015)

        model.compile(loss='mae', optimizer=simple_adam, metrics=['mae'])
        es = EarlyStopping(monitor='val_score', patience=10, verbose=2, mode='min', restore_best_weights=True, )
        es.set_model(model)
        metric = Metric(model, [es], [(X_train, y_train), (X_val, y_val)])
        model.fit(X_train, y_train, batch_size=b_size, epochs=max_epochs,
                  validation_data=[X_val, y_val],
                  callbacks=[reduce_lr], shuffle=True, verbose=2)
        y_pred3 = model.predict(X_val)
        y_pred = np.zeros((len(y_pred3),))
        sub['price'] += model.predict(test).reshape(-1, ) / n_splits
        for i in range(len(y_pred3)):
            y_pred[i] = y_pred3[i]

        oof_pred[val_idx] = y_pred
        val_mae = mean_absolute_error(y[val_idx], y_pred)
        avg_mae += val_mae / n_splits
        print()
        print('val_mae is:{}'.format(val_mae))
        print()

    mean_absolute_error(y, oof_pred)
    output_path = nn_data_path
    # 测试集输出
    sub.to_csv(output_path + 'nn_test.csv', index=False)

    # 训练集输出
    sub1 = pd.DataFrame()
    sub1['SaleID'] = pd.read_csv(train_path, sep=' ')['SaleID']
    sub1['price'] = oof_pred
    sub1.to_csv(output_path + 'nn_train.csv', index=False)
