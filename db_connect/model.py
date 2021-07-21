"""
# Create Date: 2021/7/20
# File Name : model.py
# Project NameProductAnalysis
# File Name : model
# Writer by :yhlin
"""


class Model:
    _id: str
    data: str

    def __init__(self, id: str, data: str):
        self._id = id
        self.data = data
