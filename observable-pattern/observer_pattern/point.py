'''
# Create Date: 2021/7/20 
# File Name : point.py	
# Project NameProductAnalysis	
# File Name : point	
# Writer by :yhlin	
'''
from db_connect import QueryDb
from observer_pattern import Observer, OOC


class Point(Observer):
    def update(self, observable, object_id = 0):
        if isinstance(observable, OOC):
            point = float(observable.model.data)
            if point > observable.ucl or point < observable.lcl:
                q = QueryDb()
                q.update(observable.model._id,1)
                print(f"OOC:{observable.model._id}")
