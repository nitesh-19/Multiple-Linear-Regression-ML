import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample


def linear_equation(x, w, b):
    return np.dot(x, w) + b


def model(w, b, alpha, cost_var, iterations):
    models_dict = {"Slope": w, "Y-Intercept": b, "Alpha": alpha, "Cost": cost_var, "Number of iterations": iterations,
                   }
    with open("multiple_regression_models_log.txt", "a") as file:
        file.write(str(models_dict) + "\n")
    print(f"Model: {models_dict}")


class LinearRegression:
    def __init__(self, ALPHA=0.001, ITERATIONS_LIMIT=10000, ITERATION_SAMPLING_VALUE=500, create_test_set=False):
        self.length_of_x = None
        self.ALPHA = ALPHA
        self.ITERATIONS_LIMIT = ITERATIONS_LIMIT
        self.ITERATION_SAMPLING_VALUE = ITERATION_SAMPLING_VALUE
        self.create_test_set = create_test_set
        self.working_data = None
        self.DATA_PATH = None
        self.x_index_list = None
        self.columns = None
        self.y_index = None
        self.test_set = None
        self.w = None
        self.x = None
        self.cost = None
        self.b = 0
        self.m = None
        self.should_scale_data = True

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

    def scale_data(self):
        for i in range(0, len(self.columns)):
            self.working_data[self.columns[i]] = self.working_data[self.columns[i]] / self.working_data[
                self.columns[i]].max()

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

            self.columns = [key for key in data if key in x_values]
            if y_value in data.columns:
                self.columns.append(y_value)

            self.working_data = pd.DataFrame(data=data, columns=self.columns).copy()
            if self.create_test_set is True:
                self.working_data = self.test_set_creator(self.working_data)
        self.length_of_x = len(x_values)
        self.w = np.zeros(self.length_of_x)
        if self.should_scale_data is True:
            self.scale_data()

        self.m = len(self.working_data)

        return self.working_data

    def cost_function(self, w, b):
        summation = 0
        for i in range(0, self.m):
            self.create_array(self.x, i)
            y_cap = linear_equation(self.x, self.w, self.b)
            bracket = (y_cap - self.working_data.iloc[i][0]) / (2 * self.m)
            summation += bracket
        self.cost = summation

    # for index in range(0, self.m):
    #     self.x = np.array([self.working_data.iloc[index][0]])
    #     print(self.x)

    def gradient_descent(self):
        for j in range(0, self.length_of_x):
            summation_w = 0
            summation_b = 0
            for i in range(0, self.m):
                self.create_array(self.x, i)
                y_cap = linear_equation(self.x, self.w, self.b)
                summation_w += (y_cap - self.working_data.iloc[i][-1]) * self.x[0][j]
                summation_b += y_cap - self.working_data.iloc[i][-1]
            summation_w /= self.m
            summation_b /= self.m
            temp_w = self.w[j] - (self.ALPHA * summation_w)
            temp_b = self.b - (self.ALPHA * summation_b)
            self.w[j] = temp_w
            self.b = temp_b
            self.cost_function(w=self.w, b=self.b)

    def create_array(self, char, i=0):
        if char is self.x:
            self.x = np.array([self.working_data.iloc[i][0:-1]])
        if char is self.w:
            # return np.array([self.w)
            pass

    def plot(self):

        x1_coordinate = self.working_data[self.columns[0]].min()
        y1_coordinate = linear_equation(x1_coordinate, self.w, self.b)
        print(self.w, self.b)
        x2_coordinate = self.working_data[self.columns[0]].max()
        y2_coordinate = linear_equation(x2_coordinate, self.w, self.b)

        print(self.working_data)
        print(f"Slope: {self.w}")

        self.working_data.plot.scatter(x=self.columns[0], y=self.columns[-1])
        plt.plot([x1_coordinate, x2_coordinate],
                 [y1_coordinate, y2_coordinate])
        plt.show()

    def run_trainer(self):
        self.get_training_data()
        # self.length_of_x = len
        prev_cost = None
        no_of_iterations = 0
        while no_of_iterations < self.ITERATIONS_LIMIT:
            self.gradient_descent()
            no_of_iterations += 1
            if no_of_iterations % self.ITERATION_SAMPLING_VALUE == 0:
                print(f"{no_of_iterations}/{self.ITERATIONS_LIMIT} iterations completed")
                print(self.cost)
            if self.cost == prev_cost:
                # plot()
                break
            elif no_of_iterations == self.ITERATIONS_LIMIT - 1:
                # plot()
                pass
            prev_cost = self.cost
        model(self.w, self.b, self.ALPHA, self.cost, no_of_iterations)
        self.plot()
