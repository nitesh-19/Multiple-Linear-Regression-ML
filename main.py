import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE_PATH = r".\data\world_population.csv"
COUNTRY = "China"
ALPHA = 0.001
COST_LIMIT = 1
ITERATIONS_LIMIT = 100000
DECIMAL_PRECISION = 4
SCALE_POPULATION = 10 ** 6
ITERATION_SAMPLING_VALUE = 500

cost_var = 1000

# Create a Dataframe with only target Year and Population
data = pd.read_csv(DATA_FILE_PATH)
row_india = data.loc[data["Country/Territory"] == COUNTRY].copy()

population_india = row_india.loc[:, "2022 Population":"1970 Population"].reset_index(drop=True).copy()
list_of_population = list(population_india.iloc[0]).copy()
list_of_year = [int(index.replace(" Population", "")) for index in population_india.columns]

my_data = pd.DataFrame(data=list_of_year, columns=["Year"]).copy()
my_data["Population"] = list_of_population.copy()

# Converting population to small numbers for easy calculation
my_data["Population"] = round(my_data["Population"].copy() / SCALE_POPULATION, DECIMAL_PRECISION)

# Converting Year to small numbers for easy calculation
offset_years = my_data["Year"][len(my_data) - 1]
my_data["Year"] = round(my_data["Year"] - offset_years)
print(my_data)

m = len(my_data)
w = 12
b = 5
models_dict = {}


def line_function(x, w, b):
    y = w * x + b
    return y


def cost_function(w, b):
    global cost_var
    summation = 0
    for i in range(0, m):
        y_cap = line_function(my_data.Year[i], w, b)
        y_cap = round(y_cap, DECIMAL_PRECISION)
        summation += (y_cap - my_data.Population[i]) ** 2
        summation = round(summation, DECIMAL_PRECISION)
    cost = summation / (2 * m)
    cost = round(cost, DECIMAL_PRECISION)
    cost_var = cost
    return cost


def gradient_descent():
    global w
    global b
    global no_of_iterations
    summation_w = 0
    summation_b = 0
    for i in range(0, m):
        y_cap = line_function(my_data.Year[i], w, b)
        y_cap = round(y_cap, DECIMAL_PRECISION)
        summation_w += (y_cap - my_data.Population[i]) * my_data.Year[i]
        summation_w = round(summation_w, DECIMAL_PRECISION)
    for i in range(0, m):
        y_cap = line_function(my_data.Year[i], w, b)
        y_cap = round(y_cap, DECIMAL_PRECISION)
        summation_b += (y_cap - my_data.Population[i])
        summation_b = round(summation_b, DECIMAL_PRECISION)
    summation_w /= m
    summation_b /= m
    w_temp = w - ALPHA * summation_w
    w_temp = round(w_temp, DECIMAL_PRECISION)
    b_temp = b - ALPHA * summation_b
    b_temp = round(b_temp, DECIMAL_PRECISION)
    w = w_temp
    b = b_temp
    cost_function(w, b)


def model(w, b, alpha, cost_var, iterations):
    global models_dict
    models_dict = {"Slope": w, "Y-Intercept": b, "Alpha": alpha, "Cost": cost_var, "Number of iterations": iterations,
                   "Country": COUNTRY}
    with open("models_log.txt", "a") as file:
        file.write(str(models_dict) + "\n")
    print(f"Model: {models_dict}")
    print(f"Cost: {cost_var}")


def plot():
    x1_coordinate = my_data.Year[len(my_data) - 1]
    y1_coordinate = line_function(x1_coordinate, w, b)
    x2_coordinate = my_data.Year[0]
    y2_coordinate = line_function(x2_coordinate, w, b)

    print(my_data)
    print(f"Slope: {w}")
    my_data["Population"] = round(my_data["Population"].copy() * SCALE_POPULATION, DECIMAL_PRECISION)
    my_data["Year"] = round(my_data["Year"].copy() + offset_years, DECIMAL_PRECISION)

    my_data.plot.scatter(x="Year", y="Population", ylabel="Population in Millions")
    plt.plot([x1_coordinate + offset_years, x2_coordinate + offset_years],
             [y1_coordinate * SCALE_POPULATION, y2_coordinate * SCALE_POPULATION])
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
