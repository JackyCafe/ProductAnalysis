import numpy as np

from db_connect.query_db import QueryDb
from db_connect.spc import SPC
import matplotlib.pyplot as plt


def main():
    q = QueryDb()
    param = '射出系統壓力峰值(bar)'
    cycle_time = np.array(q.get(param),dtype='float')
    spc = SPC(cycle_time)
    x_bar = spc.x_bar
    ucl = spc.ucl
    lcl = spc.lcl
    fig, axs = plt.subplots(1, figsize=(15, 15), sharex=True)
    axs.plot(x_bar, marker='o', color='black')
    axs.plot(ucl, linestyle='dashed', marker='o', color='red')
    axs.plot(lcl, linestyle='dashed', marker='o', color='red')
    axs.plot(cycle_time,linestyle='-',marker='o',color='blue')


    fig.show()


if __name__ == '__main__':
    main()
