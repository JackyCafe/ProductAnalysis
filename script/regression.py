"""
# Create Date: 2021/7/21
# File Name : regression.py
# Project NameProductAnalysis
# File Name : regression
# Writer by :yhlin
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
                    '保壓時間(秒)','延遲加料時間(秒)'], axis=1)
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
    mean = train_vaildation_data.mean()
    std = train_vaildation_data.std()
    train_data = (train_data - mean) / std
    val_data = (vaildation_data - mean) / std
    x_train = np.array(train_data.drop('NG', axis='columns'))
    y_train = np.array(train_data['NG'])
    x_test = np.array(test_data.drop('NG', axis='columns'))
    y_test = np.array(test_data['NG'])
    x_val = np.array(val_data.drop('NG', axis='columns'))
    y_val = np.array(val_data['NG'])

    model = keras.Sequential(name='model-1')
    model.add(layers.Dense(32, activation='relu', input_shape=(18,)))
    # model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.summary()
    model.compile(loss='mse',optimizer=Adam(lr=0.001),metrics=['mae'])

    model_dir = os.path.join('lab2-log', 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = os.path.join('lab2-log', 'model-1')
    model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
    model_mckp = keras.callbacks.ModelCheckpoint(
        model_dir + '/Best-model-1.h5',
        monitor='val mean absolute_err',
        save_best_only=True,
        mode='min'
    )

    history = model.fit(x_train, y_train, batch_size=32,
                        epochs=300,
                        validation_data=(x_val, y_val),
                        callbacks=[model_cbk, model_mckp]),

    test_loss,test_mae = model.evaluate(x_test,y_test)
    print(f'loss:{test_loss} ,mae:{test_mae}')

    plt.plot(history[0].history['loss'], label='train')
    plt.plot(history[0].history['val_loss'], label='vaildation')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.savefig('../pic/vaildation-relu.png')
    plt.show()
    print("done")



if __name__ == '__main__':
    main()
