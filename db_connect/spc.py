import math

import numpy as np


class SPC:
    param: np.array
    x_bar: list = []
    _ucl: list = []
    _lcl: list = []

    def __init__(self, param: np.array):
        self.param = param
        count = self.param.shape[0]
        # print(count)
        self.x_bar = np.repeat(self.mean, count)
        ucl = np.around(self.mean - 3*self.std,2)
        lcl = np.around(self.mean + 3 * self.std,2)
        print(ucl)
        print(lcl)
        self._ucl = np.repeat(ucl, count)
        self._lcl = np.repeat(lcl, count)

    @property
    def mean(self):
        return self.param.mean()

    @property
    def std(self):
        return self.param.std()

    @property
    def ucl(self) -> list:
        return self._ucl

    @property
    def lcl(self) -> list:
        return self._lcl
