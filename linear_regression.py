import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample


class LinearRegression:
    def __init__(self, ALPHA=0.001, ITERATIONS_LIMIT=10000, ITERATION_SAMPLING_VALUE=500, create_test_set=0):
        self.ALPHA = ALPHA
        self.ITERATIONS_LIMIT = ITERATIONS_LIMIT
        self.ITERATION_SAMPLING_VALUE = ITERATION_SAMPLING_VALUE
        self.create_test_set = create_test_set
        self.DATA_PATH = None
        self.x_index_list = None
        self.y_index = None
        self.test_set = None

    def test_set_creator(self, data, percent_of_data=20):
        length_of_data = len(data)

        # list_of_random_index = [319, 277, 405, 79, 129, 164, 501, 72, 369, 170, 349, 238, 358, 57, 476, 155, 389, 154,
        #                         430, 1, 306, 211, 58, 222, 433, 293, 378, 138, 16, 359, 390, 152, 283, 224, 25, 150,
        #                         226, 457, 438, 149, 523, 60, 499, 511, 93, 146, 371, 53, 339, 253, 61, 230, 243, 105,
        #                         294, 370, 46, 397, 244, 402, 208, 354, 410, 478, 41, 374, 414, 325, 252, 514, 524, 51,
        #                         196, 116, 431, 239, 122, 332, 304, 251, 26, 28, 109, 89, 262, 347, 417, 20, 449, 321,
        #                         331, 80, 91, 183, 452, 443, 212, 409, 199, 432, 219, 161, 117, 135, 133, 147, 263, 482,
        #                         483]

        list_of_random_index = sample(range(0, length_of_data), round(length_of_data * percent_of_data / 100))
        test_set = pd.DataFrame(data=data, index=list_of_random_index)
        data.drop(index=list_of_random_index, inplace=True)
        self.test_set = test_set
        return data

    def get_training_data(self):
        if self.DATA_PATH is None:
            raise TypeError(
                f"LinearRegression.DATA_PATH empty. Try setting LinearRegression.DATA_PATH to a string before starting the training.")
        if self.x_index_list is None:
            raise TypeError(
                f"LinearRegression.x_index_list empty. Try setting LinearRegression.x_index_list to a list before starting the "
                f"training.")
        if self.y_index is None:
            raise TypeError(
                f"LinearRegression.y empty. Try setting LinearRegression.y to a list before starting the "
                f"training.")
        else:
            data = pd.read_csv(self.DATA_PATH)
            x_values = [data.columns[index] for index in self.x_index_list]
            y_value = data.columns[self.y_index]

            columns = [key for key in data if key in x_values]
            if y_value in data.columns:
                columns.append(y_value)

            working_data = pd.DataFrame(data=data, columns=columns).copy()
            if self.create_test_set == 1:
                working_data = self.test_set_creator(working_data)

            return working_data
