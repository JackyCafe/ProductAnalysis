'''
# Create Date: 2021/7/20 
# File Name : observable.py	
# Project NameProductAnalysis	
# File Name : observable	
# Writer by :yhlin	
'''


class Observable:

    def __init__(self):
        self.observers = []

    def add_observer(self,observer):
        self.observers.append(observer)

    def remove_observer(self,observer):
        self.observers.remove(observer)

    def notify(self):
        for o in self.observers:
            o.update(self)