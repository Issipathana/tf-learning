from test.linear_regressor.constants import random_state, mu_X, sd_X, sd_e, B, N_samples


def my_input_fn():
    X = random_state.normal(loc=mu_X, scale=sd_X, size=N_samples)
    e = random_state.normal(loc=0, scale=sd_e, size=N_samples)
    y = X * B + e
    return X, y