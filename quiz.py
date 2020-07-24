import numpy as np
import pandas as pd

data = {'prodID': ['101', '102', '103', '104', '104'],'prodname': ['a', 'b', 'c', 'd', 'e'],'profit': ['2738', '2727', '3497', '7347', '3743']}
dataframe = pd.DataFrame(data)
dataframe
grouped_data = dataframe.groupby('prodID')
grouped_data.max()
