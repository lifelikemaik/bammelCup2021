import pandas as pd
from datetime import datetime
import csv
from sklearn.ensemble import RandomForestRegressor

import numpy as np


def format_date(s: str) -> int:
    if '-' in s:
        return int(datetime.timestamp(datetime.strptime(s, "%Y-%m-%d %H:%M:%S")))
    else:
        return int(datetime.timestamp(datetime.strptime(s, "%d.%m.%Y %H:%M")))

# Read data
result_customers = pd.read_csv('pub_f6Xd8II.csv', names=['ID', 'PREDICTION'], header=None, skiprows=1)
geo = pd.read_csv('geo.csv', names=['COUNTRY', 'SALES_OFFICE', 'SALES_BRANCH', 'SALES_LOCATION'], header=None, skiprows=1)  # 48 Standorte
transactions = pd.read_csv('transactions.csv', names=['MO_ID', 'SO_ID', 'CUSTOMER', 'END_CUSTOMER', 'OFFER_PRICE',
                                                      'SERVICE_LIST_PRICE', 'MATERIAL_COST', 'SERVICE_COST',
                                                      'PRICE_LIST',
                                                      'ISIC', 'MO_CREATED_DATE', 'SO_CREATED_DATE', 'TECH',
                                                      'OFFER_TYPE',
                                                      'BUSINESS_TYPE', 'COSTS_PRODUCT_A', 'COSTS_PRODUCT_B',
                                                      'COSTS_PRODUCT_C',
                                                      'OFFER_STATUS', 'COSTS_PRODUCT_D', 'COSTS_PRODUCT_E',
                                                      'SALES_LOCATION', 'TEST_SET_ID'], header=None, skiprows=1)

customers = pd.read_csv('customers.csv',
                        names=['CUSTOMER', 'REV_CURRENT_YEAR', 'REV_CURRENT_YEAR.1', 'REV_CURRENT_YEAR.2',
                               'CREATION_YEAR', 'OWNERSHIP', 'COUNTRY', 'CURRENCY'], header=None, skiprows=1)

transactions["CUSTOMER"] = transactions["CUSTOMER"].map(lambda x: x.lstrip('"""').rstrip('"""'))

geo["COUNTRY"] = geo["COUNTRY"].map(lambda x: x.replace("CH", "Switzerland").replace("FR", "France"))


# Clean Transactions:
transactions.fillna(0)
transactions["CUSTOMER"] = transactions["CUSTOMER"].map(lambda x: 0 if x == "NA" or x == "#NV" else x)
transactions["CUSTOMER"] = transactions["CUSTOMER"].astype(np.int64)
transactions['END_CUSTOMER'] = transactions['END_CUSTOMER'].fillna(0)
transactions['ISIC'] = transactions['ISIC'].fillna(0)

transactions['OFFER_STATUS'] = transactions['OFFER_STATUS'].fillna(0)
transactions['SALES_LOCATION'] = transactions['SALES_LOCATION'].fillna(0)

# Clean Customers:
customers.fillna(0)
customers['REV_CURRENT_YEAR'] = customers['REV_CURRENT_YEAR'].str.replace('"', '')
customers['REV_CURRENT_YEAR'] = pd.to_numeric(customers['REV_CURRENT_YEAR'], errors='coerce')

# Clean Dates
transactions["MO_CREATED_DATE"] = transactions["MO_CREATED_DATE"].map(lambda x: format_date(x))
transactions["SO_CREATED_DATE"] = transactions["SO_CREATED_DATE"].map(lambda x: format_date(x))

## Left join transactions with geodata
trans_geo = pd.merge(transactions, geo, how="left", left_on=['SALES_LOCATION'], right_on=['SALES_LOCATION'])

## Left join customer with transaction (customer id, country)

all = pd.merge(trans_geo, customers, how="left", left_on=['CUSTOMER', 'COUNTRY'], right_on=['CUSTOMER', 'COUNTRY'])

## Remove all the Test datasets, because they need to be predicted in the future

all["OFFER_STATUS"] = all["OFFER_STATUS"].map(lambda x: 1 if str(x).strip().lower()[0] == 'w' else 0)

## Encoding

all["OWNERSHIP"] = all["OWNERSHIP"].map(lambda x: 1 if str(x) == "Governmental" else 0)

all["COUNTRY"] = all["COUNTRY"].map(lambda x: 1 if str(x) == "Switzerland" else 0)

all["ISIC"] = all["ISIC"].map(lambda x: 0 if str(x) == "NA" else x)


# Encode more variables with One hot encoding

all = all.drop(["MO_ID", "SO_ID", "END_CUSTOMER", "CURRENCY", "SALES_BRANCH", "SALES_LOCATION", "PRICE_LIST", "TECH",
                "OFFER_TYPE","SALES_OFFICE", "BUSINESS_TYPE", "CREATION_YEAR", "REV_CURRENT_YEAR", "REV_CURRENT_YEAR.1", "REV_CURRENT_YEAR.2"], axis=1)

test = all[pd.to_numeric(all["TEST_SET_ID"], errors="coerce").notnull()]

all = all[pd.to_numeric(all["TEST_SET_ID"], errors="coerce").isnull()]

test_ids = test["TEST_SET_ID"]

all = all.drop("TEST_SET_ID", axis=1)

test = test.drop("TEST_SET_ID", axis=1)


## Model training

X = all.drop('OFFER_STATUS', axis=1)
y = all["OFFER_STATUS"]

test_set = test.drop("OFFER_STATUS", axis=1)

#
# Split train and test set

model = RandomForestRegressor()

model.fit(X, y)
predictions = model.predict(test_set)

result = []

counter = 0

tiddict = {}

for p in predictions:
    dict[test_ids.loc[counter]] = round(p)
    counter = counter + 1

data = pd.read_csv("pub_f6Xd8II.csv")
ids = data['id'].tolist()
header = ['id', 'prediction']

with open('predictions_the_r_tists_1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for i in ids:
        data = [i, tiddict[i]]
        writer.writerow(data)


