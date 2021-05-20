import os

import  pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Taipei Sans TC Beta']
path = os.getcwd()
workspaces = os.walk(path)
def PPA():
    df: DataFrame = pd.read_csv('data/data-2019-lot1.csv')
    keys = df.keys()
    params = keys.array
    paramCount = keys.array.shape[0]
    os.chdir("pic")
    for i in range(4,paramCount):
        for j in range(5, 6):
            mean_y = df[params[j]].mean()
            std_y = df[params[j]].std()
            y =(df[params[j]]-mean_y)/std_y
            to_draw(df,params[i],params[j])


def to_draw(df,param1,param2):

    try:
        x = np.array(df[param1])
        y = np.array(df[param2])

        coff = np.corrcoef(x,y)[0]
        fig,ax = plt.subplots()
        title =f'{param1.replace("/","_")} vs {param2.replace("/","_")} coff {coff}'
        plt.title(title)
        ax.scatter(x,y,s=10,c='red',marker='o',alpha=0.5,label = title )

        folderpath = f'{param1.replace("/","_")}'

        if not  os.path.isdir(folderpath):
            print(f"not created {folderpath}")

            os.makedirs(folderpath)
        plt.savefig(f'{folderpath}\\{title}.png')
        plt.close(fig)
    except Exception as e:
        print(e.__str__())
        print(f'except: pic\\{param1} vs{param2}')

if __name__ == '__main__':
    PPA()