# importing module
from pandas import *

data = read_csv("pub_f6Xd8II.csv")
ids = data['id'].tolist()

print('ids: ', ids)