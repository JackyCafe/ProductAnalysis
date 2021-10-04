import numpy as np

from db_connect.query_db import QueryDb
from db_connect.spc import SPC
import matplotlib.pyplot as plt

''' SPC chart 主程式'''
def main():
    q = QueryDb()
    # ids =[]
    ids = {}
    for id in q.id:
         ids[id]=0

    # param = '射出系統壓力峰值(bar)'
    # cycle_times = np.array(q.get(param)['result'], dtype='float')
    #
    # spc = SPC(cycle_times)
    # x_bar = spc.x_bar
    # ucl = spc.ucl
    # lcl = spc.lcl
    for col in list(q.cols)[5:]:
        param = np.array(q.get(col),dtype=float)
        spc = SPC(param)
        x_bar = spc.x_bar
        ucl = spc.ucl
        lcl = spc.lcl
        for id, data in zip(q.get(col)['id'],np.array(q.get(col)['result'], dtype='float')):
            if data > ucl[0] or data < lcl[0]:
                ids[id] += 1
    #             print(lcl[0],ucl[0],data)
    print([ids])


if __name__ == '__main__':
    main()
