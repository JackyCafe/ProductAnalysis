'''
# Create Date: 2021/7/20 
# File Name : observer_pattern.py
# Project NameProductAnalysis	
# File Name : observer_pattern
# Writer by :yhlin	
'''
import abc


class Observer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def update(self, observable ,object_id = 0):
        pass
