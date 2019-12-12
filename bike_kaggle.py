import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('ds_bike/train.csv', parse_dates=["datetime"])
train.shape
test = pd.read_csv('ds_bike/test.csv', parse_dates=["datetime"])
test.shape
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["hour"] = train["datetime"].dt.hour
train["dayofweek"] = train["datetime"].dt.dayofweek
train.shape
test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["hour"] = test["datetime"].dt.hour
test["dayofweek"] =test["datetime"].dt.dayofweek
test.shape
categorical_feature_names = ["season", "holiday", "workingday", "weather", "dayofweek", "month", "year", "hour"]
for var in categorical_feature_names:
    train[var] = train[var].astype("category")
    test[var] = test[var].astype("category")
feature_names = ["season", "weather", "temp", "atemp", "humidity", "year", "hour", "dayofweek", "holiday", "workingday"]
feature_names
X_train = train[feature_names]
print(X_train.shape)
X_train.head()
X_test = test[feature_names]
print(X_test.shape)
X_test.head()
label_name = "count"
y_train = train[label_name]
print(y_train.shape)
y_train.head()
from sklearn.metrics import make_scorer
def rmsle(predicted_values, actual_values, convertExp=True):
    if convertExp:
            predicted_values = np.exp(predicted_values),
            actual_values = np.exp(actual_values)
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)
    difference = log_predict - log_actual
    difference = np.square(difference)
    mean_difference = difference.mean()
    score = np.sqrt(mean_difference)
    return score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
lModel = LinearRegression()
y_train_log = np.log1p(y_train)
lModel.fit(X_train, y_train_log)
preds = lModel.predict(X_train)
print("RMSLE Values for Linear Regression: ", rmsle(np.exp(y_train_log), np.exp(preds), False))
ridge_m_ = Ridge()
ridge_params_ = {'max_iter': [3000], 'alpha':[0.01, 0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000]}
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)
grid_ridge_m = GridSearchCV(ridge_m_, ridge_params_, scoring = rmsle_scorer, cv=5)
y_train_log = np.log1p(y_train)
grid_ridge_m.fit(X_train, y_train_log)
preds = grid_ridge_m.predict(X_train)
print(grid_ridge_m.best_params_)
print("RMSLE Value For Ridge Regression: ", rmsle(np.exp(y_train_log), np.exp(preds), False))
df = pd.DataFrame(grid_ridge_m.cv_results_)
df.head()
df["alpha"] = df["params"].apply(lambda x: x["alpha"])
df["rmsle"] = df["mean_test_score"].apply(lambda x:-x)
df[["alpha", "rmsle"]].head()
fig, ax = plt.subplots()
fig.set_size_inches(12,5)
plt.xticks(rotation=30, ha='right')
sns.pointplot(data=df, x="alpha", y="rmsle", ax=ax)
plt.show()
lasso_m_=Lasso()
alpha = 1/np.array([0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000])
lasso_params_ = {'max_iter':[3000], 'alpha':alpha}
grid_lasso_m = GridSearchCV(lasso_m_, lasso_params_, scoring=rmsle_scorer, cv=5)
y_train_log = np.log1p(y_train)
grid_lasso_m.fit(X_train, y_train_log)
preds = grid_lasso_m.predict(X_train)
print(grid_lasso_m.best_params_)
print("RMSLE Value For Lasso Regression: ", rmsle(np.exp(y_train_log), np.exp(preds), False))
df = pd.DataFrame(grid_lasso_m.cv_results_)
df["alpha"] = df["params"].apply(lambda x:x["alpha"])
df["rmsle"] = df["mean_test_score"].apply(lambda x: -x)
df[["alpha", "rmsle"]].head()
fig, ax = plt.subplots()
fig.set_size_inches(12, 5)
plt.xticks(rotation=30, ha='right')
sns.pointplot(data=df, x="alpha", y="rmsle", ax=ax)
plt.show()
from sklearn.ensemble import RandomForestRegressor
rfModel = RandomForestRegressor(n_estimators=100)
y_train_log = np.log1p(y_train)
rfModel.fit(X_train, y_train_log)
preds = rfModel.predict(X_train)
score = rmsle(np.exp(y_train_log), np.exp(preds), False)
print("RMSLE Value For Random Forest: " , score)
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=4000, alpha=0.01);
y_train_log = np.log1p(y_train)
gbm.fit(X_train, y_train_log)
preds = gbm.predict(X_train)
gbm.fit(X_train, y_train_log)
score = rmsle(np.exp(y_train_log), np.exp(preds), False)
print("RMSLE Value For Gradient Boost: ", score)
predsTest = gbm.predict(X_test)
fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.set_size_inches(12, 5)
sns.distplot(y_train, ax=ax1, bins=50)
sns.distplot(np.exp(predsTest), ax=ax2, bins=50)
plt.show()
# submission = pd.read_csv("ds_bike/sampleSubmission.csv")
# submission["count"] = np.exp(predsTest)
# print(submission.shape)
# submission.head()
# submission.to_csv(f"Score_{score:.5f}_submission.csv", index=False)
