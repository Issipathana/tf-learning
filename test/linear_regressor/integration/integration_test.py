import tensorflow as tf

from linear_regressor.estimator import LinearRegressor
from linear_regressor.model_functions import my_model_fn


estimator = LinearRegressor(my_model_fn)
