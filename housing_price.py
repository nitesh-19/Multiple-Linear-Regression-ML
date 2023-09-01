import pandas as pd
import matplotlib.pyplot as plt


ALPHA = 0.01
COST_LIMIT = 1
ITERATIONS_LIMIT = 10000
ITERATION_SAMPLING_VALUE = 500
DATA_FILE_PATH = r".\data\Housing.csv"
data = pd.read_csv(DATA_FILE_PATH)
# print(data)
# list_price = data["price"]
my_data = pd.DataFrame(columns=["Price", "Area"])
my_data["Price"] = data.price / 100000
my_data["Area"] = data.area / 1000
m = len(my_data)
w = 12
b = 5
cost_var = 1000000
no_of_iterations = 0


def line_function(x, w, b):

    y = w * x + b
    return y


def cost_function(w, b):
    global cost_var
    summation = 0
    for i in range(0, m):
        y_cap = line_function(my_data.Area[i], w, b)
        bracket = ((y_cap - my_data.Price[i]) ** 2) / (2 * m)
        bracket = ((y_cap - my_data.Price[i]) ** 2) / (2 * m)
        summation += bracket
    cost_var = summation


def gradient_descent():
    global w
    global b
    global no_of_iterations
    summation_w = 0
    summation_b = 0
    for i in range(0, m):
        y_cap_1 = line_function(my_data.Area[i], w, b)
        bracket_1 = ((y_cap_1 - my_data.Price[i]) * my_data.Area[i]) / m
        summation_w += bracket_1

    for i in range(0, m):
        y_cap = line_function(my_data.Area[i], w, b)
        bracket = (y_cap - my_data.Price[i]) / m
        summation_b += bracket

    w_temp = w - ALPHA * summation_w
    # w_temp = round(w_temp, DECIMAL_PRECISION)
    b_temp = b - ALPHA * summation_b
    # b_temp = round(b_temp, DECIMAL_PRECISION)
    w = w_temp
    b = b_temp
    cost_function(w, b)


def model(w, b, alpha, cost_var, iterations):
    models_dict = {"Slope": w, "Y-Intercept": b, "Alpha": alpha, "Cost": cost_var, "Number of iterations": iterations,
                   }
    with open("models_log.txt", "a") as file:
        file.write(str(models_dict) + "\n")
    print(f"Model: {models_dict}")


def plot():
    global w
    global b
    global ALPHA
    global cost_var
    global no_of_iterations

    x1_coordinate = my_data.Area[len(my_data) - 1]
    y1_coordinate = line_function(x1_coordinate, w, b)
    print(w, b)
    x2_coordinate = my_data.Area[0]
    y2_coordinate = line_function(x2_coordinate, w, b)

    print(my_data)
    print(f"Slope: {w}")

    my_data.plot.scatter(x="Area", y="Price")
    plt.plot([x1_coordinate, x2_coordinate],
             [y1_coordinate, y2_coordinate])
    plt.show()
    model(w, b, ALPHA, cost_var, no_of_iterations)


if __name__ == "__main__":
    prev_cost = None
    no_of_iterations = 0
    while cost_var > COST_LIMIT and no_of_iterations < ITERATIONS_LIMIT:
        gradient_descent()
        no_of_iterations += 1
        if no_of_iterations % ITERATION_SAMPLING_VALUE == 0:
            print(f"{no_of_iterations}/{ITERATIONS_LIMIT} iterations completed")
            print(cost_var)
        if prev_cost == cost_var:
            plot()
            break
        elif no_of_iterations == ITERATIONS_LIMIT - 1:
            plot()
        prev_cost = cost_var
