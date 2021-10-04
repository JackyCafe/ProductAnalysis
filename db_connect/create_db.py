import pandas as pd
from pandas import DataFrame
from pymongo import MongoClient


class CreateDB:
    df: DataFrame

    def __init__(self, filename: str):
        self.df = pd.read_csv(filename)
        self.clients = MongoClient()
        self.database = self.clients['product_db']
        self.collection = self.database['product_data']
        self.cols = self.df.columns.array
        self.records = self.df.to_dict("records")

    def insert(self):
        self.collection.insert_many(self.records)


def main():
    db = CreateDB('../data/data-2019-lot1.csv')
    db.insert()
    print('done')


if __name__ == '__main__':
    main()
