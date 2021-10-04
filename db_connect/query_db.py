import pandas as pd
from pymongo import MongoClient
from bson.json_util import dumps

from db_connect.model import Model


class QueryDb:
    def __init__(self):
        self.clients = MongoClient()
        self.database = self.clients['product_db']
        self.collection = self.database['product_data']
        self.datas = self.collection.find({},{})

    @property
    def data(self):
        cursor = self.datas
        return list(cursor)

    def __len__(self):
        return self.collection.count_documents({})

    def __getitem__(self, position: int) -> dict:
        return self.datas[position]

    @property
    def cols(self):
        dataset = self.collection
        return dataset.find_one({}).keys()

    @property
    def id(self):
        datas = self.collection.find({}, {'_id': 1})
        return [f'{data.get("_id")}' for num, data in enumerate(datas, start=1)]

    def get(self, param: str) -> list:
        id = []
        result = []
        cursors = self.collection.find({}, {param})

        for cursor in cursors:
            id = cursor['_id']
            data = cursor[param]
            model = Model(id, data)
            result.append(model)
        return result

    def get_data_to_df(self):
        df = pd.DataFrame(list(self.collection.find({})))
        return df

    def mean(self, param: str) -> float:
        result = self.get(param)
        sum: float = 0.0
        count: int = len(result.get(param))
        for x in result.get(param):
            sum += float(x)
        return sum / count

    def update(self, id: str, value: int):
        self.collection.update_one({"_id": id}, {"$set": {"NG": value}})

    def update_pred(self, id: str, value: float):
        self.collection.update_one({"_id": id}, {"$set": {"Pred": value}})
