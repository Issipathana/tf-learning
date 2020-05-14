from linear_regressor.input_functions import my_input_fn

X, y = my_input_fn()
print(X)
print(type(X), X.shape, X.dtype)
