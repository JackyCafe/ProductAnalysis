"""
# Create Date: 2021/10/3 
# File Name : draw_chart.py	
# Project NameProductAnalysis	
# File Name : draw_chart	
# Writer by :yhlin	
"""
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from db_connect import QueryDb


def main():
    q = QueryDb()

    df = q.get_data_to_df()
    x_data = df.drop(['_id', '資料時間', '工單編號', '產品名稱', '產品編號',
                    '模穴壓力峰值(1)(bar)', '重量(1)', '重量(2)',
                    '保壓時間(秒)', '延遲加料時間(秒)'], axis=1)
    data = df.drop(['_id', '資料時間', '工單編號', '產品名稱', '產品編號',
                    '模穴壓力峰值(1)(bar)', '重量(1)', '重量(2)',
                    '保壓時間(秒)', '延遲加料時間(秒)'], axis=1)
    x_test = np.array(data.drop('NG', axis='columns'))
    mean = x_test.mean()
    std = x_test.std()
    x_test = (x_test-mean)/std
    data = df.get([ '射出系統壓力峰值(bar)','射出速度峰值(mm/s)', '第三段料溫(℃)', '第四段料溫(℃)', '第五段料溫(℃)', '第六段料溫(℃)','NG'])

    model = keras.Sequential(name='model-1')
    model.add(layers.Dense(32, activation='sigmoid', input_shape=(18,)))
    # model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(32, activation='sigmoid'))
    model.add(layers.Dense(1))
    model.summary()
    # model.built = True

    model.load_weights('../lab2-log/models/Best-model-1.h5')
    y_pred = model.predict(x_test).flatten()
    print(y_pred)
    mean = data.mean()
    std = data.std()
    x = (data-mean)/std
    x1 = x['射出系統壓力峰值(bar)']
    x2 = x['射出速度峰值(mm/s)']
    x3 = x['第三段料溫(℃)']
    x4 = x['第四段料溫(℃)']
    x5 = x['第五段料溫(℃)']#, , ]
    x6 = x['第六段料溫(℃)']
    plt.plot(y_pred,'y')
    plt.show()
    plt.scatter(y_pred,x1, label='射出系統壓力峰值(bar)')
    plt.scatter(y_pred,x2, label='射出速度峰值(mm/s)')
    plt.scatter(y_pred,x3, label='射出速度峰值(mm/s)')
    plt.scatter(y_pred, x4, label='射出速度峰值(mm/s)')
    plt.scatter(y_pred, x5, label='射出速度峰值(mm/s)')
    plt.scatter(y_pred, x6, label='射出速度峰值(mm/s)')

    plt.show()

if __name__ == '__main__':
    main()