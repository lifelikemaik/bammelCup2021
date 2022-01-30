import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

seed = 2022

def format_date(s: str) -> int:
    if '-' in s:
        return int(datetime.timestamp(datetime.strptime(s, "%Y-%m-%d %H:%M:%S")))
    else:
        return int(datetime.timestamp(datetime.strptime(s, "%d.%m.%Y %H:%M")))

result_customers = pd.read_csv('pub_f6Xd8II.csv')
geo = pd.read_csv('geo.csv')  # 48 Standorte
transactions = pd.read_csv('transactions.csv')

customers = pd.read_csv('customers.csv')

transactions["CUSTOMER"] = transactions["CUSTOMER"].map(lambda x: x.lstrip('"""').rstrip('"""'))

geo["COUNTRY"] = geo["COUNTRY"].map(lambda x: x.replace("CH", "Switzerland").replace("FR", "France"))
transactions["CUSTOMER"] = transactions["CUSTOMER"].map(lambda x: 0 if x == "NA" or x == "#NV" else x)
transactions["CUSTOMER"] = transactions["CUSTOMER"].astype(np.int64)
transactions['END_CUSTOMER'] = transactions['END_CUSTOMER'].fillna(0)
transactions['ISIC'] = transactions['ISIC'].fillna(0)
transactions['SALES_LOCATION'] = transactions['SALES_LOCATION'].fillna(0)
customers.fillna(0)
customers['REV_CURRENT_YEAR'] = customers['REV_CURRENT_YEAR'].str.replace('"', '')
customers['REV_CURRENT_YEAR'] = pd.to_numeric(customers['REV_CURRENT_YEAR'], errors='raise')
transactions["MO_CREATED_DATE"] = transactions["MO_CREATED_DATE"].map(lambda x: format_date(x))
transactions["SO_CREATED_DATE"] = transactions["SO_CREATED_DATE"].map(lambda x: format_date(x))
geo = geo[geo['SALES_BRANCH'].notna()]
geo = geo[geo['SALES_LOCATION'].notna()]
geo = geo[geo['SALES_OFFICE'].notna()]

trans_geo = pd.merge(transactions, geo, how="left", left_on=['SALES_LOCATION'], right_on=['SALES_LOCATION'])
all = pd.merge(trans_geo, customers, how="left", left_on=['CUSTOMER', 'COUNTRY'], right_on=['CUSTOMER', 'COUNTRY'])
all = all[all["OFFER_STATUS"] != "NA"]
train_df1 = all[all['TEST_SET_ID'].isna()].copy()
train_df1.drop('TEST_SET_ID', inplace=True, axis=1)
test_df1 = all[all['TEST_SET_ID'].notna()].copy()
train_df1["OFFER_STATUS"] = train_df1["OFFER_STATUS"].map(lambda x: 1 if str(x).strip().lower()[0] == 'w' else 0)

categorical_cols = [
    'BUSINESS_TYPE', 'SALES_OFFICE', 'SALES_LOCATION', 'TECH', 'OFFER_TYPE', 'ISIC',
    'PRICE_LIST', 'OWNERSHIP', 'COUNTRY', 'CURRENCY'
]

for col in categorical_cols:
    train_df1[col] = train_df1[col].astype(str)
    test_df1[col] = test_df1[col].astype(str)

df = train_df1
encoder = OneHotEncoder(handle_unknown='ignore')
rev_cols = ['REV_CURRENT_YEAR', 'REV_CURRENT_YEAR.1', 'REV_CURRENT_YEAR.2']

rev_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
)

col_transformer = ColumnTransformer(
    transformers=[
        ("cat", encoder, categorical_cols),
        ("rev", rev_transformer, rev_cols)
    ]
)

df = df.drop([
    "MO_ID", "SO_ID", "CREATION_YEAR", "CUSTOMER",
    "END_CUSTOMER", "SALES_BRANCH"
], axis=1)

test_df1.drop([
"MO_ID", "SO_ID", "CREATION_YEAR", "CUSTOMER",
    "END_CUSTOMER", "SALES_BRANCH"
], axis=1, inplace=True)

X = df.drop('OFFER_STATUS', axis=1)
y = df["OFFER_STATUS"]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

forest = RandomForestClassifier(random_state=seed)

clf = Pipeline(
    steps=[("preprocessor", col_transformer), ("classifier", forest)]
)
clf.fit(train_X, train_y)
y_pred = clf.predict(val_X)
res = clf.predict(test_df1.drop(["OFFER_STATUS", "TEST_SET_ID"], axis=1))
res2 = test_df1[['TEST_SET_ID']].astype(int).rename(columns={'TEST_SET_ID': 'id'})
res2['prediction'] = res
res2.to_csv('predictions_the_r_tists_1.csv', index=False)