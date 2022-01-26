import joblib
import pandas as pd
from datetime import datetime

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

import numpy as np


def format_date(s: str) -> int:
    if '-' in s:
        return int(datetime.timestamp(datetime.strptime(s, "%Y-%m-%d %H:%M:%S")))
    else:
        return int(datetime.timestamp(datetime.strptime(s, "%d.%m.%Y %H:%M")))

# Read data
# https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html NA rausballern mit read_csv
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


print(customers.head())

print(transactions["CUSTOMER"].tail(20))

# https://stackoverflow.com/questions/21491291/remove-all-quotes-within-values-in-pandas

# Clean Transactions:
# TODO further cleaning (?)
#transactions = transactions.drop(transactions[(transactions.TEST_SET_ID).isnull()].index)
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

print(trans_geo["COUNTRY"])

print(customers["COUNTRY"])


## Left join customer with transaction (customer id, country)

all = pd.merge(trans_geo, customers, how="left", left_on=['CUSTOMER', 'COUNTRY'], right_on=['CUSTOMER', 'COUNTRY'])

print(all)

## Remove all the Test datasets, because they need to be predicted in the future
# all = all.drop(all[all."OFFER_STATUS" == 'NA'].index)
# all = all[all["OFFER_STATUS"] != "NA"]

print(all["OFFER_STATUS"])

all["OFFER_STATUS"] = all["OFFER_STATUS"].map(lambda x: 1 if str(x).strip().lower()[0] == 'w' else 0)


print(all["OFFER_STATUS"])

## Remove Columns that are not needed

# Encode more variables with One hot encoding
all = all.drop(["MO_ID", "SO_ID", "END_CUSTOMER", "CURRENCY", "SALES_BRANCH", "SALES_LOCATION", "PRICE_LIST", "TECH",
                "OFFER_TYPE","SALES_OFFICE", "BUSINESS_TYPE", "CREATION_YEAR", "REV_CURRENT_YEAR", "REV_CURRENT_YEAR.1", "REV_CURRENT_YEAR.2"], axis=1)

test = all[pd.to_numeric(all["TEST_SET_ID"], errors="coerce").notnull()]

all = all[pd.to_numeric(all["TEST_SET_ID"], errors="coerce").isnull()]

all = all.drop("TEST_SET_ID", axis=1)

test = test.drop("TEST_SET_ID", axis=1)

## Encoding

all["OWNERSHIP"] = all["OWNERSHIP"].map(lambda x: 1 if str(x) == "Governmental" else 0)

all["COUNTRY"] = all["COUNTRY"].map(lambda x: 1 if str(x) == "Switzerland" else 0)

# all["END_CUSTOMER"] = all["END_CUSTOMER"].map(lambda x: 0 if str(x) == "NA" else x)

all["ISIC"] = all["ISIC"].map(lambda x: 0 if str(x) == "NA" else x)


## Model training
# TODO implement granularity -> suboffer

X = all.drop('OFFER_STATUS', axis=1)
y = all["OFFER_STATUS"]



#
# Split train and test set
#
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


model = RandomForestRegressor()

model.fit(train_X, train_y)
predictions = model.predict(val_X)



#
# Error-Estimation
#
right = 0
wrong = 0
nums = 0

print(type(predictions))
print(type(val_y))

for p in predictions:
    if round(p) == val_y.iloc[nums]:
        right = right + 1
    else:
        wrong = wrong + 1

    nums = nums + 1

print("Richtig: " + str(right))
print("Falsch: " + str(wrong))
print("Insgesamt:" + str(wrong/nums))

joblib.dump(predictions, "./random_forest.joblib")



