#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os


# In[11]:


import tarfile


# In[12]:


import urllib


# In[13]:


DOWNLOAD_ROOT =  "https://raw.githubusercontent.com/ageron/handson-ml2/master/"


# In[14]:


HOUSING_PATH = os.path.join("datasets/housing/housing.tgz")


# In[15]:


HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# In[16]:


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[17]:


fetch_housing_data()


# In[60]:


import pandas as pd


# In[19]:


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[20]:


housing = load_housing_data()
housing.head()


# In[21]:


housing.info()


# In[22]:


housing["ocean_proximity"].value_counts()


# In[23]:


housing.describe()


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


import matplotlib.pyplot as plt


# In[26]:


housing.hist(bins=50, figsize=(20,15))


# In[27]:


plt.show()


# In[28]:


import numpy as np


# In[29]:


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[30]:


train_set, test_set = split_train_test(housing, 0.2)


# In[31]:


len(train_set)


# In[32]:


len(test_set)


# In[33]:


from zlib import crc32


# In[34]:


def test_set_check(identified, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


# In[35]:


def split_train_test_by_id(data, test_ratio, id_column):
 ids = data[id_column]
 in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
 return data.loc[~in_test_set], data.loc[in_test_set]


# In[38]:


housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# In[39]:


housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[43]:


housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])


# In[44]:


housing["income_cat"].hist()


# In[45]:


import numpy as np


# In[46]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[47]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[48]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
# In[49]:


housing = strat_train_set.copy()


# In[50]:


housing.plot(kind="scatter", x="longitude", y="latitude")


# In[51]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[53]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
            )
plt.legend()


# In[62]:


from pandas.plotting import scatter_matrix


# In[63]:


attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[64]:


housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


# In[98]:


#Experimenting with Attribute Coimbinations
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[65]:


housing = strat_train_set.drop("median_house_value", axis=1) # Prep the Data for ML Algorithms. REvert to clean training set. drop() creates a copy of the data and doesn't affect strat_train_set
housing_labels = strat_train_set["median_house_value"].copy()


# In[66]:


# Data Cleaning. Fixing missing values
from sklearn.impute import SimpleImputer


# In[67]:


imputer = SimpleImputer(strategy="median")


# In[68]:


# the median can only be computed on numerical attributes, therefore create a copy of the data without the text attribute ocean_proximity
housing_num = housing.drop("ocean_proximity", axis=1)


# In[69]:


imputer.fit(housing_num)


# In[71]:


imputer.statistics_


# In[74]:


housing_num.median().values


# In[75]:


X = imputer.transform(housing_num)


# In[76]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing_num.index)


# In[78]:


# Handling Text and Categorical Attributes
housing_cat = housing[["ocean_proximity"]]


# In[80]:


housing_cat.head(10)


# In[81]:


# most ML algorithms prefer to work with numbers
# Convert these categories from text to numbers using Scikit-Learn OrdinalEncoder class
from sklearn.preprocessing import OrdinalEncoder


# In[82]:


ordinal_encoder = OrdinalEncoder()


# In[84]:


housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)


# In[85]:


housing_cat_encoded[:10]


# In[87]:


ordinal_encoder.categories_


# In[88]:


# binary attribute per category
# one-hot encoding. Scikit-Learn OneHotEncoder class converts categorical values into one-hot vectors.
from sklearn.preprocessing import OneHotEncoder


# In[89]:


cat_encoder = OneHotEncoder()


# In[90]:


housing_cat_1hot = cat_encoder.fit_transform(housing_cat)


# In[91]:


housing_cat_1hot


# In[92]:


# output is SciPy sparse matrix instead of NumPy array. Useful for categorical attributes with thousands of categories. 
# Sparse matrix only stores the location of non-zero elements, saving memory usage


# In[ ]:


#transformations
# fit the scalers to the training data only, not the full dataset including the test set. Then yoiu can use them to transform the training set and the test set (and new data)


# In[93]:


# Normalization: min-max scaling. Values are shifted and rescaled to a range from 0 to 1.
# Standardization: doesn't bind values to a specific range, potentially problematic for some algorithms. Sdzn is less affected by outliers.


# In[104]:


# Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin


# In[105]:


rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


# In[110]:


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                        bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# In[111]:


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[94]:


# Transformation Pipelines
# order is important.
from sklearn.pipeline import Pipeline


# In[95]:


from sklearn.preprocessing import StandardScaler


# In[112]:


num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()),
 ])


# In[113]:


housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[114]:


# use single transformer to handle all categorical and numerical columns, and apply appropriate transformations to each column.
from sklearn.compose import ColumnTransformer


# In[116]:


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]


# In[117]:


full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])


# In[118]:


housing_prepared = full_pipeline.fit_transform(housing)


# In[119]:


# Preprocessing pipeline that includes the full housing data and applies the appropriate transformations to each column has been set up.
# Framed the problem
# got the data and explored it
#sampled a training set and a test set
# wrote transformation pipelines to clean up and prepare the data for ML algorithms automatically


# In[120]:


# Select and Train a Machine Learning Model


# In[121]:


# Training and Evaluating on the Training Set
from sklearn.linear_model import LinearRegression


# In[122]:


lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[123]:


some_data = housing.iloc[:5]


# In[124]:


some_labels = housing_labels.iloc[:5]


# In[125]:


some_data_prepared = full_pipeline.transform(some_data)


# In[126]:


print("Predictions:", lin_reg.predict(some_data_prepared))


# In[127]:


print("Labels:", list(some_labels))


# In[128]:


from sklearn.metrics import mean_squared_error


# In[129]:


housing_predictions = lin_reg.predict(housing_prepared)


# In[130]:


lin_mse = mean_squared_error(housing_labels, housing_predictions)


# In[131]:


lin_rmse = np.sqrt(lin_mse)


# In[132]:


lin_rmse


# In[133]:


# fix underfitting by: select more powerful model, feed training algorithm with better features, or reduce the constraints on the model.


# In[134]:


# Option 1: Try a more complex Model
# DecisionTreeRegressior; powerful model, ca-able of finding complex non-linear relationships in the data.
from sklearn.tree import DecisionTreeRegressor


# In[135]:


tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[136]:


housing_predictions = tree_reg.predict(housing_prepared)


# In[137]:


tree_mse = mean_squared_error(housing_labels, housing_predictions)


# In[138]:


tree_rmse = np.sqrt(tree_mse)


# In[139]:


tree_rmse


# In[140]:


# no error in model? likely cause: model has badly overfit the data. NExt step is to check this and confirm


# In[141]:


# Better Evaluation Using Cross-Validation:
# to evaluate the Decision Tree model, use the train_test_split function to split the training set into a smaller training set and a validation set, then train your models against the smaller set and evaluate them against the validation set


# In[142]:


# Scikit-Learn K-fold cross-validation
# following code randomely spolits the training set into 10 subsets called folds.
# it then trains and evaluates the Decision Tree model 10 times, picking a different fold evaluation every time and training on the other 9 folds.
# result: array containing the 10 evaluation scores


# In[143]:


from sklearn.model_selection import cross_val_score


# In[144]:


scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[145]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[146]:


display_scores(tree_rmse_scores)


# In[148]:


# now compute the same scores for the Linear Regression model to be sure:
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                            scoring="neg_mean_squared_error", cv=10)


# In[149]:


lin_rmse_scores = np.sqrt(-lin_scores)


# In[151]:


display_scores(lin_rmse_scores)


# In[152]:


# Try one more model
# RandomForestRegressor
# Random Forests work by training many Decisioon Trees on random subsets of the features, then averaging out their predictions.
# Ensemble learning: building a model on top of another model.


# In[153]:


from sklearn.ensemble import RandomForestRegressor


# In[155]:


forest_reg = RandomForestRegressor()


# In[156]:


forest_reg.fit(housing_prepared, housing_labels)


# In[161]:


scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)


# In[163]:


forest_rmse_scores


# In[177]:


forest_rmse = np.sqrt(forest_rmse)


# In[178]:


forest_rmse


# In[164]:


#Fine-Tune the Model
from sklearn.model_selection import GridSearchCV


# In[175]:


param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]


# In[166]:


forest_reg = RandomForestRegressor()


# In[167]:


grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
 scoring='neg_mean_squared_error',
return_train_score=True)


# In[168]:


grid_search.fit(housing_prepared, housing_labels)


# In[169]:


grid_search.best_params_


# In[170]:


cvres = grid_search.cv_results_


# In[172]:


for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[173]:


feature_importances = grid_search.best_estimator_.feature_importances_


# In[174]:


feature_importances


# In[176]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]


# In[179]:


cat_encoder = full_pipeline.named_transformers_["cat"]


# In[180]:


cat_one_hot_attribs = list(cat_encoder.categories_[0])


# In[181]:


attributes = num_attribs + extra_attribs + cat_one_hot_attribs


# In[182]:


sorted(zip(feature_importances, attributes), reverse=True)


# In[183]:


# Evaluate your system on the Test Set
final_model = grid_search.best_estimator_


# In[184]:


X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()


# In[185]:


X_test_prepared = full_pipeline.transform(X_test)


# In[186]:


final_predictions = final_model.predict(X_test_prepared)


# In[187]:


final_mse = mean_squared_error(y_test, final_predictions)


# In[188]:


final_rmse = np.sqrt(final_mse)


# In[190]:


# compute a 95 % confidence interval for ther generalization error
from scipy import stats


# In[191]:


confidence = 0.95


# In[192]:


squared_errors = (final_predictions - y_test) ** 2


# In[193]:


np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc=squared_errors.mean(),
                        scale=stats.sem(squared_errors)))


# In[194]:


import joblib


# In[198]:


joblib.dump(ml_project_Housing_Prices_Cali, "ml_project_Housing_Prices_Cali.pkl")


# In[ ]:




