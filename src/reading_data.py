import pandas as pd
from io import StringIO


class ReadingData:

    def reading_data(self, file_name, num_symbols):
        f = open(file_name, "r")
        data = f.read(num_symbols)
        data1 = io.StringIO(data)
        df = pd.read_csv(data1, sep="\t", header=None)
        df1 = df.rename(columns={0: 'target', 1: 'text'})
        df1['target'] = df1['target'].map({'spam': 1, 'ham': 0})

        return df1
