from numpy import array

import joblib
import pandas as pd
from datetime import datetime

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import linear_model

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



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


#print(customers.head())

#print(transactions["CUSTOMER"].tail(20))

# https://stackoverflow.com/questions/21491291/remove-all-quotes-within-values-in-pandas

# Clean Transactions:
# TODO clean end_customer

#transactions = transactions.drop(transactions[(transactions.TEST_SET_ID).isnull()].index)
#transactions.fillna(0)
transactions["CUSTOMER"] = transactions["CUSTOMER"].map(lambda x: 0 if x == "NA" or x == "#NV" else x)
transactions["CUSTOMER"] = transactions["CUSTOMER"].astype(np.int64)
transactions['END_CUSTOMER'] = transactions['END_CUSTOMER'].fillna(0)
transactions['ISIC'] = transactions['ISIC'].fillna(0)

#print(transactions["END_CUSTOMER"])

transactions['OFFER_STATUS'] = transactions['OFFER_STATUS'].fillna(0)
transactions['SALES_LOCATION'] = transactions['SALES_LOCATION'].fillna(0)

# Clean Customers:
customers.fillna(0)
customers['REV_CURRENT_YEAR'] = customers['REV_CURRENT_YEAR'].str.replace('"', '')
customers['REV_CURRENT_YEAR'] = pd.to_numeric(customers['REV_CURRENT_YEAR'], errors='coerce')

# Clean Dates
transactions["MO_CREATED_DATE"] = transactions["MO_CREATED_DATE"].map(lambda x: format_date(x))
transactions["SO_CREATED_DATE"] = transactions["SO_CREATED_DATE"].map(lambda x: format_date(x))

# Clean geo.csv
geo = geo[geo['SALES_BRANCH'].notna()]
geo = geo[geo['SALES_LOCATION'].notna()]
geo = geo[geo['SALES_OFFICE'].notna()]



#geo.to_csv(r'~/AnalyticsCup/pyramidProject/export_geo.csv', index=False, header=True)

## Left join transactions with geodata
trans_geo = pd.merge(transactions, geo, how="left", left_on=['SALES_LOCATION'], right_on=['SALES_LOCATION'])

#print(trans_geo["COUNTRY"])

#print(customers["COUNTRY"])


## Left join customer with transaction (customer id, country)

all = pd.merge(trans_geo, customers, how="left", left_on=['CUSTOMER', 'COUNTRY'], right_on=['CUSTOMER', 'COUNTRY'])


## Remove all the Test datasets, because they need to be predicted in the future
all = all[all["OFFER_STATUS"] != "NA"]


all["OFFER_STATUS"] = all["OFFER_STATUS"].map(lambda x: 1 if str(x).strip().lower()[0] == 'w' else 0)


# Remove Columns that are not needed

# Encode more variables with hot one encoding
# Hot one encoding
# TODO verify encoding
categorical_cols = ['BUSINESS_TYPE', 'SALES_BRANCH', 'SALES_LOCATION', 'TECH', 'OFFER_TYPE']
df = pd.get_dummies(all, columns=categorical_cols)

#print("\n\nOne hot encoding\n")
#print(df)

# TODO encode mo_id, so_id, END_CUSTOMER,CURRENCY,SALES_BRANCH
all = all.drop(["MO_ID", "SO_ID", "SALES_LOCATION", "OFFER_TYPE", "SALES_OFFICE",
                "CREATION_YEAR", "REV_CURRENT_YEAR", "REV_CURRENT_YEAR.1", "REV_CURRENT_YEAR.2"], axis=1)

test = all[pd.to_numeric(all["TEST_SET_ID"], errors="coerce").notnull()]

all = all[pd.to_numeric(all["TEST_SET_ID"], errors="coerce").isnull()]

all = all.drop("TEST_SET_ID", axis=1)

test = test.drop("TEST_SET_ID", axis=1)
print(len(test))
## Encoding

all["OWNERSHIP"] = all["OWNERSHIP"].map(lambda x: 1 if str(x) == "Governmental" else 0)

all["COUNTRY"] = all["COUNTRY"].map(lambda x: 1 if str(x) == "Switzerland" else 0)

all["ISIC"] = all["ISIC"].map(lambda x: 0 if str(x) == "NA" else x)

all["END_CUSTOMER"] = all["END_CUSTOMER"].map(lambda x: 10000 if str(x) == "NA"
    else (10001 if str(x) == "No"
          else (10002 if str(x) == "Yes"
                else x)))

all["CURRENCY"] = all["CURRENCY"].map(lambda x: 0 if str(x) == "Euro"
    else (1 if str(x) == "US Dollar"
          else (2 if str(x) == "Pound Sterling"
                else 3)))
# 0 = Euro; 1 = US Dollar; 2 = Pound Sterling; 3 = Chinese Yuan

all["PRICE_LIST"] = all["PRICE_LIST"].map(lambda x: 0 if str(x) == "CMT End Customer"
    else (1 if str(x) == "CMT Installer"
          else (2 if str(x) == "SFT Standard"
                else 3)))
# 0 = CMT End Customer; 1 = CMT Installer; 2 = SFT Standard; 3 = Traffic public

all["TECH"] = all["TECH"].map(lambda x: 0 if str(x) == "BP"
    else (1 if str(x) == "C"
          else (2 if str(x) == "F"
                else 3)))
# 0 = BP; 1 = C; 2 = F; 3 = S

all["BUSINESS_TYPE"] = all["BUSINESS_TYPE"].map(lambda x: 0 if str(x) == "C"
    else (1 if str(x) == "E"
          else (2 if str(x) == "Exp"
                else (3 if str(x) == "M"
                      else (4 if str(x) == "Mig"
                            else (5 if str(x) == "N"
                                   else (6 if str(x) == "New"
                                         else (7 if str(x) == "R"
                                               else (8 if str(x) == "S"
                                                     else 9)))))))))
# 0 = C; 1 = E; 2 = Exp; 3 = M; 4 = Mig; 5 = N; 6 = New; 7 = R; 8 = S; 9 = T

all["SALES_BRANCH"] = all["SALES_BRANCH"].map(lambda x: 0 if str(x) == "Branch Central"
    else (1 if str(x) == "Branch East"
          else (2 if str(x) == "Branch West"
                else (3 if str(x) == "Centre-Est"
                      else (4 if str(x) == "Enterprise Business France"
                            else (5 if str(x) == "EPS CH"
                                  else (6 if str(x) == "Grand Est"
                                        else (7 if str(x) == "Grand Paris"
                                              else (8 if str(x) == "Nord FR"
                                                    else (9 if str(x) == "Quest"
                                                          else (10 if str(x) == "SI"
                                                                else (11 if str(x) == "Sud Quest"
                                                                      else (12)))))))))))))
# 0 = Branch Central; 1 = Branch East; 2 = Branch West; 3 = Centre-Est;
# 4 = Enterprise Business France; 5 = EPS CH; 6 = Grand Est; 7 = Grand Paris;
# 8 = Nord FR; 9 = Quest; 10 = SI; 11 = Sud Quest; 12 = Sud-Est


# "OFFER_TYPE", "SALES_OFFICE","SALES_LOCATION" hat zu viele variablen um das per hand zu machen = OnehOtencoding am besten

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


print(list(all.columns))
print(list(test.columns))

forest = RandomForestClassifier()
forest.fit(train_X, train_y)
y_pred = forest.predict(val_X)

print(accuracy_score(val_y, y_pred))

for p in predictions:
    if round(p) == val_y.iloc[nums]:
        right = right + 1
    else:
        wrong = wrong + 1
    nums = nums + 1




print("Richtig: " + str(right))
print("Falsch: " + str(wrong))
print("Insgesamt:" + str(wrong/nums))

# TODO generate .csv with the results
#data.to_csv(r'~/AnalyticsCup/pyramidProject/export_model.csv')
#joblib.dump(df, "./random_forest.joblib")



