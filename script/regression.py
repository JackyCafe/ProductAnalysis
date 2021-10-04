"""
# Create Date: 2021/7/21
# File Name : regression.py
# Project NameProductAnalysis
# File Name : regression
# Writer by :yhlin
# 跑線性回歸的程式
"""
import os
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from tensorflow.python.keras.optimizer_v2.adam import Adam

from db_connect import QueryDb
import matplotlib.pyplot as plt


def main():
    q = QueryDb()
    df = q.get_data_to_df()
    data = df.drop(['_id', '資料時間', '工單編號', '產品名稱', '產品編號',
                    '模穴壓力峰值(1)(bar)', '重量(1)', '重量(2)',
                    '保壓時間(秒)', '延遲加料時間(秒)'], axis=1)
    # data = df.drop(['_id', '資料時間', '工單編號', '產品名稱', '產品編號',
    #                 '模穴壓力峰值(1)(bar)', '重量(1)', '重量(2)',
    #                 '保壓時間(秒)','延遲加料時間(秒)'], axis=1)
    # 捨棄一些文字結構的欄位

    # 資料先打亂
    data_num = data.shape[0]
    indexes = np.random.permutation(data_num)
    train_index = indexes[: int(data_num * 0.6)]
    vaildation_index = indexes[int(data_num * 0.6):int(data_num * 0.8)]
    test_index = indexes[int(data_num * 0.8):]
    train_data = data.loc[train_index]
    vaildation_data = data.loc[vaildation_index]
    test_data = data.loc[test_index]
    train_vaildation_data = pd.concat([train_data, vaildation_data])
    y_train = np.array(train_data['NG'])
    y_test = np.array(test_data['NG'])
    y_val = np.array(vaildation_data['NG'])

    mean = train_vaildation_data.mean()
    std = train_vaildation_data.std()
    train_data = (train_data - mean) / std
    val_data = (vaildation_data - mean) / std
    test_data = (test_data-mean)/std
    x_train = np.array(train_data.drop('NG', axis='columns'))
    x_test = np.array(test_data.drop('NG', axis='columns'))
    columns = data.columns.drop('NG')
    out= pd.DataFrame(x_test,columns=columns)

    x_val = np.array(val_data.drop('NG', axis='columns'))

    model = keras.Sequential(name='model-1')
    model.add(layers.Dense(32, activation='sigmoid', input_shape=(18,)))
    # model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(32, activation='sigmoid'))
    model.add(layers.Dense(1))
    model.summary()
    model.compile(loss='mse',optimizer=Adam(lr=0.001),metrics=['mae','acc'])

    model_dir = os.path.join('../lab2-log', 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = os.path.join('../lab2-log', 'model-1')
    model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
    model_mckp = keras.callbacks.ModelCheckpoint(
        model_dir + '/Best-model-1.h5',
        monitor='val_mean_absolute_err',
        save_best_only=True,
        mode='min'
    )

    history = model.fit(x_train, y_train, batch_size=32,
                        epochs=1000,
                        validation_data=(x_val, y_val),
                        callbacks=[model_cbk, model_mckp]),

    y_pred = model.predict(x_test).flatten()
    plt.plot(y_pred,'y')

    plt.show()
    # for x,y in zip(x_test,y_pred):
    #     print(x,y)

    '''todo  將ypred 寫到db'''
    acc = keras.metrics.binary_accuracy(y_test,y_pred)
    print(f'acc:{acc}')
    test_loss,test_mae,acc = model.evaluate(x_test,y_test)
    print(f'loss:{test_loss} ,mae:{test_mae}')


    plt.plot(history[0].history['loss'], label='train')
    plt.plot(history[0].history['val_loss'], label='vaildation')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.savefig('../pic/vaildation-relu.png')
    plt.show()
    print("done")

    df = q.get_data_to_df()

    data = df.drop(['_id', '資料時間', '工單編號', '產品名稱', '產品編號',
                    '模穴壓力峰值(1)(bar)', '重量(1)', '重量(2)',
                    '保壓時間(秒)', '延遲加料時間(秒)'], axis=1)
    x_test = np.array(data.drop('NG', axis='columns'))
    # print(data.head())
    mean = x_test.mean()
    std = x_test.std()
    x_test = (x_test-mean)/std
    # data = df.get([ '射出系統壓力峰值(bar)','射出速度峰值(mm/s)', '第三段料溫(℃)', '第四段料溫(℃)', '第五段料溫(℃)', '第六段料溫(℃)','NG'])

    y_pred = model.predict(x_test).flatten()
    print(y_pred)
    mean = data.mean()
    std = data.std()
    x = (data-mean)/std
    x1 = x['射出系統壓力峰值(bar)']
    x2 = x['射出速度峰值(mm/s)']
    x3 = x['第三段料溫(℃)']
    x4 = x['第四段料溫(℃)']
    x5 = x['第五段料溫(℃)']
    x6 = x['第六段料溫(℃)']
    # plt.plot(y_pred,'y')
    # plt.show()
    plt.scatter(y_pred,x1, label='射出系統壓力峰值(bar)')
    plt.scatter(y_pred,x2, label='射出速度峰值(mm/s)')
    plt.scatter(y_pred,x3, label='射出速度峰值(mm/s)')


    plt.show()

if __name__ == '__main__':
    main()
