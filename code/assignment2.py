#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import scipy.sparse

import statsmodels.api as sm

# import tqdm
# import sklearn.linear_model
# import sklearn.metrics

import sklearn.linear_model
import sklearn.metrics

# score function: binary cross entropy loss
def score_yp(y, p):  # y, p are numpy arrays
    return sklearn.metrics.log_loss(y, p)


# Checking data
baskets = pd.read_parquet("../data/baskets-s.parquet")
coupons = pd.read_parquet("../data/coupons-s.parquet")

print(baskets.head())

print(coupons.head())

# INPUT
training_week = 88
validation_week = 89
test_week = 90
target_customers = list(range(2000))
target_products = list(range(250))

# Maximum weeks
print(baskets.week.max())

# Number of weeks
num_weeks = baskets.week.nunique()
print(num_weeks)

# Baseline Model, using purchase frequency to calculate baseline probability
purchase_frequency_ij = (
    (baskets.groupby(["customer", "product"])[["week"]].count() / num_weeks)
    .rename(columns={"week": "probability"})
    .reset_index()
)

print(purchase_frequency_ij)

# Compute co-occurence matrix
def co_occurrences_sparse(x, variable_basket="customer", variable_product="product"):
    row = x[variable_basket].values
    col = x[variable_product].values
    dim = (x[variable_basket].max() + 1, x[variable_product].max() + 1)

    basket_product_table = scipy.sparse.csr_matrix(
        (np.ones(len(row), dtype=int), (row, col)), shape=dim
    )
    co_occurrences_sparse = basket_product_table.T.dot(basket_product_table)
    co_occurrences_dense = co_occurrences_sparse.toarray()
    return co_occurrences_dense


co_occurrences_2 = co_occurrences_sparse(baskets)

print(co_occurrences_2)


# In[ ]:


# In[20]:


# function to define target variable for all customer-product combinations (in a given week)
def build_target(baskets, week):

    baskets_week = baskets[baskets["week"] == week][
        ["week", "customer", "product"]
    ].reset_index(drop=True)
    baskets_week["y"] = 1

    df = pd.DataFrame(
        {
            "week": week,
            "customer": np.repeat(target_customers, len(target_products), axis=0),
            "product": target_products * len(target_customers),
        }
    )

    df = df.merge(baskets_week, on=["week", "customer", "product"], how="left")
    df["y"] = df["y"].fillna(0).astype(int)

    return df


baseline_target = build_target(baskets, validation_week)
baseline_target.head()


# baseline = purchase rates for customer-product combinations before the target week
def baseline_prediction(baskets, week):

    # subset baskets
    baskets_t = baskets[baskets["week"] < week].reset_index(drop=True)
    n_weeks = baskets_t.week.nunique()
    print(n_weeks)

    # model (non-0 probabilities)
    purchase_frequency_ij = (
        (baskets_t.groupby(["customer", "product"])[["week"]].count() / n_weeks)
        .rename(columns={"week": "probability"})
        .reset_index()
    )

    # filling in 0s
    df = pd.DataFrame(
        {
            "week": week,
            "customer": np.repeat(target_customers, len(target_products), axis=0),
            "product": target_products * len(target_customers),
        }
    )

    result_baseline = pd.merge(
        df,
        purchase_frequency_ij,
        on=["customer", "product"],
        how="left",
    ).fillna(0)

    return result_baseline


# prediction for validation data
baseline_validation = baseline_prediction(baskets, validation_week)
baseline_validation.head()

# score wrapper, for data frames. we need this when using the `truth` data frame
def score(x, y):  # x, y are data frames
    xy = pd.merge(x, y, on=["customer", "product", "week"])
    assert xy.shape[0] == x.shape[0]
    return score_yp(xy["y"].values, xy["probability"].values)


# Baseline prediction
baseline_test = baseline_prediction(baskets, test_week)
print(baseline_test.head())

# Baseline prediction
print(score(baseline_target, baseline_validation))


# In[21]:


# Function for implementing the discount feature - Vid
def build_discount_feature(coupons, week):

    subset_coupons = coupons[coupons["week"] == week][
        ["customer", "product"]
    ].reset_index(drop=True)
    subset_coupons = pd.concat(
        [
            subset_coupons,
            pd.Series(coupons[coupons["week"] == week]["discount"].values / 100).rename(
                "discount"
            ),
        ],
        axis=1,
    )

    return subset_coupons


# In[22]:


# Build frequency feature
def build_frequency_feature(baskets, week_start, week_end, feature_name):
    # subset baskets
    baskets_subset = baskets[
        (baskets["week"] >= week_start) & (baskets["week"] <= week_end)
    ]
    print(baskets_subset.week.nunique())

    purchase_frequency_ij = (
        (
            baskets_subset.groupby(["customer", "product"])[["week"]].count()
            / baskets_subset.week.nunique()
        )
        .rename(columns={"week": feature_name})
        .reset_index()
    )

    return purchase_frequency_ij


# Base dataframe for logit model
def build_base_table(baskets, week, coupons):
    # target variable (product purchase)
    # consider using multiple weeks for training! more data might lead to better results.
    # also, different weeks might have different information.
    y = build_target(baskets, week)
    # features
    # note how features are computed on data BEFORE the target week
    x_1 = build_frequency_feature(baskets, -1, week - 1, "frequency_full")
    x_2 = build_frequency_feature(baskets, week - 30, week - 1, "frequency_l30")
    x_3 = build_frequency_feature(baskets, week - 5, week - 1, "frequency_l5")
    x_4 = build_frequency_feature(
        baskets, week - 13, week - 1, "frequency_3m"
    )  # Vid's code -> looking at quarters
    x_5 = build_discount_feature(coupons, week)

    base_table_yx = (
        y.merge(x_1, on=["customer", "product"], how="left")
        .merge(x_2, on=["customer", "product"], how="left")
        .merge(x_3, on=["customer", "product"], how="left")
        .merge(
            x_4, on=["customer", "product"], how="left"
        )  # Vid's code -> adding the quarters
        .merge(
            x_5, on=["customer", "product"], how="left"
        )  # Vid's code -> adding the discounts
        .fillna(0)
    )

    return base_table_yx


# In[23]:


# Building training base dataframe (week < 89)
base_table_train = build_base_table(baskets, training_week, coupons)

print(base_table_train.head())

# Logit model
y = base_table_train["y"].values  # 1s and 0s

X = base_table_train[
    ["frequency_full", "frequency_l30", "frequency_l5", "frequency_3m", "discount"]
].values  # purchase frequencies

log_reg = sklearn.linear_model.LogisticRegression().fit(X, y)

# Logit intercept + coefficients (to be interpreted)
print(log_reg.intercept_)
print(log_reg.coef_)

# use model to predict purchase probabilities
base_table_train["probability"] = log_reg.predict_proba(X)[:, 1]

print(
    score_yp(
        base_table_train["y"].values,
        base_table_train["probability"].values,
    )
)

print(base_table_train.head())


# In[24]:


# Validation
base_table_validation = build_base_table(baskets, validation_week, coupons)

X_validation = base_table_validation[
    ["frequency_full", "frequency_l30", "frequency_l5", "frequency_3m", "discount"]
].values

base_table_validation["probability"] = log_reg.predict_proba(X_validation)[:, 1]

score_yp(
    base_table_validation["y"].values,
    base_table_validation["probability"].values,
)


# In[25]:


# Linear regression
base_table_train = build_base_table(baskets, training_week, coupons)

print(base_table_train.head())

y = base_table_train["y"].values  # 1s and 0s
X = base_table_train[
    ["frequency_full", "frequency_l30", "frequency_l5", "frequency_3m", "discount"]
].values

lin_reg = sklearn.linear_model.LinearRegression().fit(X, y)

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

print(lin_reg.intercept_)
print(lin_reg.coef_)

base_table_validation["probability_linear"] = lin_reg.predict(X)

score_yp(
    base_table_validation["y"].values,
    base_table_validation["probability_linear"].values,
)


# In[26]:


# Random forest model
from sklearn.ensemble import RandomForestRegressor

base_table_train = build_base_table(baskets, training_week, coupons)

print(base_table_train.head())

y = base_table_train["y"].values  # 1s and 0s
X = base_table_train[
    ["frequency_full", "frequency_l30", "frequency_l5", "frequency_3m", "discount"]
].values

Random_forest = RandomForestRegressor(
    n_estimators=150, max_depth=50, oob_score=True, n_jobs=15, verbose=1
)

Random_forest.fit(X, y)

base_table_train["probability"] = Random_forest.predict(X)


# In[27]:


print(base_table_train.head())

score_yp(
    base_table_train["y"].values,
    base_table_train["probability"].values,
)


# In[28]:


# Validation logit

base_table_validation_logit = build_base_table(baskets, validation_week, coupons)

X_validation = base_table_validation_logit[
    ["frequency_full", "frequency_l30", "frequency_l5", "frequency_3m", "discount"]
].values

base_table_validation_logit["probability"] = log_reg.predict_proba(X_validation)[:, 1]

score_yp(
    base_table_validation_logit["y"].values,
    base_table_validation_logit["probability"].values,
)


# In[29]:


# Validation linear

base_table_validation_linear = build_base_table(baskets, validation_week, coupons)

print(base_table_validation_linear.head())

y = base_table_validation_linear["y"].values  # 1s and 0s
X = base_table_validation_linear[
    ["frequency_full", "frequency_l30", "frequency_l5", "frequency_3m", "discount"]
].values

lin_reg = sklearn.linear_model.LinearRegression().fit(X, y)

print(lin_reg.intercept_)
print(lin_reg.coef_)

base_table_validation_linear["probability_linear"] = lin_reg.predict(X)

score_yp(
    base_table_validation_linear["y"].values,
    base_table_validation_linear["probability_linear"].values,
)


# In[30]:


# Validation random forest

base_table_validation_forest = build_base_table(baskets, validation_week, coupons)

y = base_table_validation_forest["y"].values  # 1s and 0s
X = base_table_validation_forest[
    ["frequency_full", "frequency_l30", "frequency_l5", "frequency_3m", "discount"]
].values

Random_forest = RandomForestRegressor(
    n_estimators=150, max_depth=50, oob_score=True, n_jobs=15, verbose=1
)

Random_forest.fit(X, y)

base_table_validation_forest["probability"] = Random_forest.predict(X)

print(base_table_validation_forest.head(20))

score_yp(
    base_table_validation_forest["y"].values,
    base_table_validation_forest["probability"].values,
)


# In[31]:


# Predictions random forest
base_table_prediction_forest = build_base_table(baskets, test_week, coupons)

y = base_table_prediction_forest["y"].values  # 1s and 0s
X_prediction_table = base_table_validation_forest[
    ["frequency_full", "frequency_l30", "frequency_l5", "frequency_3m", "discount"]
].values


base_table_prediction_forest["probability"] = Random_forest.predict(X_prediction_table)

Random_forest.fit(X_prediction_table, y)

print(base_table_prediction_forest.head(20))


# In[ ]:
