import numpy as np
import pandas as pd
from typing import List

from utils import get_now_str


class Recorder:
    def __init__(self, header: List[str]):
        self.header = header
        self.csv_file = open("Data/data_%s.csv" % get_now_str(), 'w')
        pd.DataFrame(columns=self.header).to_csv(
            self.csv_file, encoding='utf-8', index=False, header=True)
    
    def append(self, data: List):
        data = np.hstack(data).reshape((1, len(data)))
        pd.DataFrame(data).to_csv(
            self.csv_file, encoding='utf-8', index=False, header=False)
