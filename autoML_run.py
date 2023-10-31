"""Run overnight / late"""

from tpot import TPOTRegressor
from functions import load_data, get_train_targets, get_test_data, prepare_submission
from sklearn.model_selection import train_test_split, RepeatedKFold

data_a, data_b, data_c = load_data()

drop_cols = ['time', 'date_calc', 'elevation:m', 'fresh_snow_1h:cm', 'wind_speed_u_10m:ms', 
             'wind_speed_u_10m:ms', 'wind_speed_v_10m:ms', 'wind_speed_w_1000hPa:ms']

# ------ for location A -----

X_train_a, targets_a = get_train_targets(data_a)


X = X_train_a.drop(columns=drop_cols).fillna(0)
y = targets_a

cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

model = TPOTRegressor(generations=5, population_size=50, scoring='neg_mean_absolute_error', cv=cv, verbosity=2, random_state=1, n_jobs=-1)
model.fit(X, y)
model.export('tpot_locA_best_model.py')

# ------ for location B -----

X_train_b, targets_b = get_train_targets(data_b)


X = X_train_b.drop(columns=drop_cols).fillna(0)
y = targets_b

cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

model = TPOTRegressor(generations=5, population_size=50, scoring='neg_mean_absolute_error', cv=cv, verbosity=2, random_state=1, n_jobs=-1)
model.fit(X, y)
model.export('tpot_locB_best_model.py')

# ------ for location C -----

X_train_c, targets_c = get_train_targets(data_c)

drop_cols = ['time', 'date_calc', 'elevation:m', 'fresh_snow_1h:cm', 'wind_speed_u_10m:ms', 
             'wind_speed_u_10m:ms', 'wind_speed_v_10m:ms', 'wind_speed_w_1000hPa:ms']

X = X_train_c.drop(columns=drop_cols).fillna(0)
y = targets_c

cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

model = TPOTRegressor(generations=5, population_size=50, scoring='neg_mean_absolute_error', cv=cv, verbosity=2, random_state=1, n_jobs=-1)
model.fit(X, y)
model.export('tpot_locC_best_model.py')