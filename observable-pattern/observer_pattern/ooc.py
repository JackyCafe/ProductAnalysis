'''
# Create Date: 2021/7/20 
# File Name : ooc.py
# Project NameProductAnalysis	
# File Name : OOC	
# Writer by :yhlin	
'''
from db_connect.model import Model
from observer_pattern import Observable


class OOC(Observable):
    point: float
    model: Model

    def __init__(self, lcl: float, ucl: float):
        super(OOC, self).__init__()
        self.lcl = lcl
        self.ucl = ucl

    def getPoint(self) -> float:
        return self.point

    def setPoint(self, i: int,point: float):
        self.point = point
        print(f'i:{i},lcl:{self.lcl},point:{self.point},ucl:{self.ucl}')
        self.notify()

    def setModel(self,model: Model):
        self.model = model
        print(f' lcl:{self.lcl},point:{model.data},ucl:{self.ucl}')
        self.notify()
