from pymongo import MongoClient


class QueryDb:
    def __init__(self):
        self.clients = MongoClient()
        self.database = self.clients['product_db']
        self.collection = self.database['product_data']
        self.datas = self.collection.find({})

    @property
    def data(self):
        return self.datas

    def __len__(self):
        return self.collection.count_documents({})

    def __getitem__(self, position: int)->dict:
        return self.datas[position]

    @property
    def cols(self):
        dataset = self.collection
        return dataset.find_one({}).keys()

    def get(self,param:str)->list:
        id = []
        result=[]
        datas = self.collection.find({},{param:1})
        for num,data in enumerate(datas,start=1):
            id.append(f'{data.get("_id")}')
            result.append(f'{data.get(param)}')

        return result

    def mean(self,param: str)->float:
        result = self.get(param)
        sum : float = 0.0
        count: int = len(result.get(param))
        for x in result.get(param):
            sum += float(x)
        return sum/count




