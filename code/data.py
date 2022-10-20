import os
import numpy as np
import pandas as pd

import scipy.sparse
import scipy.stats

# import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.metrics


# Data Path
DATA_PATH = "data/"

# Check if required files are available
assert os.path.isfile(f"{DATA_PATH}/baskets-s.parquet")
assert os.path.isfile(f"{DATA_PATH}/coupons-s.parquet")
assert os.path.isfile(f"{DATA_PATH}/prediction_index.parquet")

# Checking data
baskets = pd.read_parquet(f"{DATA_PATH}/baskets-s.parquet")
coupons = pd.read_parquet(f"{DATA_PATH}/coupons-s.parquet")
prediction_index = pd.read_parquet(f"{DATA_PATH}/prediction_index.parquet")

print("Customer/Baskets dataset:")

print(baskets.head())

# The first week
print(baskets["week"].min())
# The last week
print(baskets["week"].max())

# The first customer
print(baskets["customer"].min())
# The final customer
print(baskets["customer"].max())

# The first product
print(baskets["product"].min())
# The last product
print(baskets["product"].max())

# Average price (money spent)
print(baskets["price"].mean())
# Standard Deviation price (money spent)
print(baskets["price"].std())
# The minimum price
print(baskets["price"].min())
# The maximum price
print(baskets["price"].max())

print("Coupons data set:")

# The first week
print(coupons["week"].min())
# The last week
print(coupons["week"].max())

# The first customer
print(coupons["customer"].min())
# The last customer
print(coupons["customer"].max())

# The first product
print(coupons["product"].min())
# The last product
print(coupons["product"].max())

# Average discount
print(coupons["discount"].mean())
# Standard Deviation price (money spent)
print(coupons["discount"].std())
# The minimum price
print(coupons["discount"].min())
# The maximum price
print(coupons["discount"].max())

# Correlation between past purchases and coupons use.
# INPUT
training_week = 88
validation_week = 89
test_week = 90
target_customers = list(range(2000))
target_products = list(range(250))

# Number of weeks
num_weeks = baskets.week.nunique()
print(num_weeks)

# Baseline Model, using purchase frequency to calculate baseline probability
purchase_frequency_ij = (
    (baskets.groupby(["customer", "product"])[["week"]].count() / num_weeks)
    .rename(columns={"week": "probability"})
    .reset_index()
)
# print(purchase_frequency_ij)
