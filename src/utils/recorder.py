import numpy as np
import pandas as pd
from typing import List

from .common import get_now_str


class Recorder:
    def __init__(self, header: List[str]):
        self.header = header
        self.csv_file = "data/data_%s.csv" % get_now_str()
        with open(self.csv_file, 'w') as f:
            pd.DataFrame(columns=self.header).to_csv(f, index=False, header=True)
    
    def append(self, data: List):
        with open(self.csv_file, 'a') as f:
            pd.DataFrame([data]).to_csv(f, index=False, header=False)
