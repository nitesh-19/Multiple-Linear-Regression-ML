import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample


def linear_equation(x, w, b):
    return np.dot(x, w) + b


def predict_with_model(x):
    x = np.array(x)
    with open("fully_trained_models.txt", "r") as file:
        lines = file.readlines()
        parameter_dict = eval(lines[-1])
    w = np.array(eval(parameter_dict["Slope"]))
    b = np.array(eval(parameter_dict["Y - Intercept"]))
    return linear_equation(x, w, b)


class LinearRegression:
    def __init__(self, DATA_PATH, feature_columns_index, target_column_index, ALPHA=0.001, ITERATIONS_LIMIT=10000,
                 ITERATION_SAMPLING_VALUE=5, create_test_set=False, should_scale_data=True):
        self.length_of_x = None
        self.create_test_set = create_test_set
        self.working_data = None
        self.DATA_PATH = DATA_PATH
        self.x_index_list = np.array(feature_columns_index)
        self.columns = None
        self.y_index = target_column_index
        self.test_set = None
        self.w = None
        self.x = None
        self.b = 0
        self.cost = None
        self.m = None
        self.should_scale_data = should_scale_data
        self.scale_factors = []
        self.no_of_iterations = 0
        self.ITERATIONS_LIMIT = ITERATIONS_LIMIT
        self.ITERATION_SAMPLING_VALUE = ITERATION_SAMPLING_VALUE
        self.ALPHA = ALPHA
        self.start_prompt()
        self.run_trainer()

    def start_prompt(self):
        print("Do you want to resume training the last interrupted model?")
        response = input("Press 'y' to load the last model weights or press 'n' to set the weights to default: ")
        if response == "y":
            self.create_test_set = False
            self.get_last_model()
            input("Press any key to start training: ")

        elif response == "n":
            print("Setting weights to default and starting the trainer.")
        else:
            print("Invalid Input.")
            self.start_prompt()

    def get_last_model(self):
        with open("partially_trained_models.txt", "r") as file:
            lines = file.readlines()
            parameter_dict = eval(lines[-1])
        self.w = np.array(eval(parameter_dict["Slope"]))
        self.b = np.array(eval(parameter_dict["Y - Intercept"]))
        self.no_of_iterations = parameter_dict["Number of iterations"]
        # self.ALPHA = parameter_dict["Alpha"]
        print(f"Scaled w = {self.w}, b = {self.b}, Alpha = {self.ALPHA}")

    def save_model(self, file_path, w, b, alpha, cost_var, iterations):
        models_dict = {"Dataset": self.DATA_PATH, "Slope": str(w.tolist()), "Y - Intercept": str(b.tolist()),
                       "Alpha": alpha,
                       "Cost": str(cost_var),
                       "Number of iterations": iterations,
                       }
        with open(file_path, "a") as file:
            file.write(str(models_dict) + "\n")
        print(f"Model: {models_dict}")

    def test_set_creator(self, data, percent_of_data=20):
        length_of_data = len(data)

        # list_of_random_index =

        list_of_random_index = sample(range(0, length_of_data), round(length_of_data * percent_of_data / 100))
        test_set = pd.DataFrame(data=data, index=list_of_random_index)
        data.drop(index=list_of_random_index, inplace=True)
        test_set.to_csv("test_set.csv")
        data.to_csv("training_set.csv")

        self.test_set = test_set
        return data

    def scale_data(self, unscale=False):
        if not unscale:
            for i in range(0, len(self.columns)):
                self.scale_factors.append(self.working_data[self.columns[i]].max())
                self.working_data[self.columns[i]] = self.working_data[self.columns[i]] / self.working_data[
                    self.columns[i]].max()
        elif unscale:
            for i in range(0, len(self.columns)):
                self.working_data[self.columns[i]] = self.working_data[self.columns[i]] * self.scale_factors[i]
            # self.b = self.b * self.scale_factors[-1]
            for index in range(0, len(self.columns) - 1):
                self.w[index] = self.w[index] * self.scale_factors[-1] / self.scale_factors[index]

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

        if self.w is None:
            self.w = np.zeros(self.length_of_x)
        if self.should_scale_data is True:
            self.scale_data()

        self.m = len(self.working_data)

        return self.working_data

    def cost_function(self):
        summation = 0
        for i in range(0, self.m):
            self.create_array(self.x, i)
            y_cap = linear_equation(self.x, self.w, self.b)
            bracket = (y_cap - self.working_data.iloc[i][0]) / (2 * self.m)
            summation += bracket
        self.cost = np.round(summation, 16)

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
            self.cost_function()

    def create_array(self, char, i=0):
        if char is self.x:
            self.x = np.array([self.working_data.iloc[i][0:-1]])

    def plot(self):
        self.scale_data(unscale=True)
        x_coordinates = []
        y_coordinates = []

        for i in range(0, self.length_of_x):
            x1_coordinate = self.working_data[self.columns[i]].min()
            y1_coordinate = linear_equation(x1_coordinate, self.w[i], self.b)
            x2_coordinate = self.working_data[self.columns[i]].max()
            y2_coordinate = linear_equation(x2_coordinate, self.w[i], self.b)
            x_coordinates.append((x1_coordinate, x2_coordinate))
            y_coordinates.append((y1_coordinate, y2_coordinate))
            self.working_data.plot.scatter(x=self.columns[i], y=self.columns[-1])

            plt.plot(x_coordinates[i], y_coordinates[i])
        print(f"Slope: {self.w}")

        plt.show()

    def run_trainer(self):
        self.get_training_data()
        prev_cost = None
        try:
            while self.no_of_iterations < self.ITERATIONS_LIMIT:
                self.gradient_descent()
                self.no_of_iterations += 1
                if self.no_of_iterations % self.ITERATION_SAMPLING_VALUE == 0:
                    print(f"{self.no_of_iterations}/{self.ITERATIONS_LIMIT} iterations completed")
                    print(f"Current Cost: {self.cost}")
                if prev_cost is not None:
                    if round(self.cost[0], 8) == round(prev_cost[0], 8):
                        break
                prev_cost = self.cost

        except KeyboardInterrupt:
            self.save_model(file_path="partially_trained_models.txt", w=self.w, b=self.b, alpha=self.ALPHA,
                            cost_var=self.cost, iterations=self.no_of_iterations)
            self.plot()
        else:
            self.plot()
            self.save_model(file_path="fully_trained_models.txt", w=self.w, b=self.b, alpha=self.ALPHA,
                            cost_var=self.cost, iterations=self.no_of_iterations)
