'''
# Create Date: 2021/7/20 
# File Name : test.py	
# Project NameProductAnalysis	
# File Name : test	
# Writer by :yhlin	
'''

import numpy as np

from db_connect.query_db import QueryDb
from db_connect.spc import SPC
import matplotlib.pyplot as plt

from observer_pattern import OOC, Point


def main():
    q = QueryDb()

    ids = []
    # for id in q.id:
    #      ids[id]=0

    param = '射出系統壓力峰值(bar)'
    cols = list(q.cols)
    for col in cols[5:]:
        models = q.get(col)
        datas = []
        # values = np.array(datas, dtype='float')
        for i, model in enumerate(models):
            ids.append(model._id)
            datas.append(model.data)
        values = np.array(datas, dtype=float)
        spc = SPC(values)
        x_bar = spc.x_bar
        ucl = spc.ucl[0] #上管制界線
        lcl = spc.lcl[0] #下管制界線
        ooc = OOC(lcl,ucl)
        point = Point()
        ooc.add_observer(point)
        for model in models:
            ooc.setModel(model)
    #     ooc.setPoint(i,v)


if __name__ == '__main__':
    main()
