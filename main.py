import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE_PATH = r".\data\world_population.csv"
COUNTRY = "India"

# Create Dataframe with only Year and Population
data = pd.read_csv(DATA_FILE_PATH)
row_india = data.loc[data["Country/Territory"] == COUNTRY]

population_india = row_india.loc[:, "2022 Population":"1970 Population"].reset_index(drop=True)
list_of_population = list(population_india.iloc[0])
list_of_year = [int(index.replace(" Population", "")) for index in population_india.columns]

my_data = pd.DataFrame(data=list_of_year,
                       columns=["Year"])
my_data["Population"] = list_of_population
my_data["Population"] = round(my_data["Population"] * 10 ** -8)  # Converting population to small numbers
print(my_data)
# my_data["Year"] = round(my_data["Year"] - my_data["Year"][len(my_data)-1])  # Converting Year to small numbers
# subtract_value = my_data["Year"][0] - my_data["Year"][len(my_data)-1]
# print(subtract_value)
print(my_data)

# def cost_function():
cost_var = 1000
m = len(my_data)
w = 5
b = 10
alpha = 0.001
models = []


def line_function(x, w, b):
    y = w * x + b
    return y


def cost_function(w, b):
    global cost_var
    summation = 0
    for i in range(0, m):
        y_cap = line_function(my_data.Year[i], w, b)
        temp_summation = int(y_cap - my_data.Population[i]) * int(y_cap - my_data.Population[i])
        summation += int(temp_summation)
        int(summation)
    if summation > 10000000000:
        summation *= 10000000000000000000
        round(summation, 10000)
        int(summation)
    cost = int(summation // (2 * m))
    cost_var = cost
    return cost


def gradient_descent():
    global w
    global b
    summation_w = 0
    summation_b = 0
    for i in range(0, m):
        y_cap = line_function(my_data.Year[i], w, b)
        summation_w += (y_cap - my_data.Population[i]) * my_data.Year[i]
        int(summation_w)
    for i in range(0, m):
        y_cap = line_function(my_data.Year[i], w, b)
        summation_b += (y_cap - my_data.Population[i])
        int(summation_b)
    summation_w /= m
    int(summation_w)
    summation_b /= m
    int(summation_b)
    w_temp = int(w - alpha * summation_w)

    b_temp = int(b - alpha * summation_b)
    w = w_temp
    b = b_temp
    cost_function(w, b)


def model(w, b, alpha, cost_var):
    global models
    models.append({"Slope": w, "Y-Intercept": b, "Alpha": alpha, "Cost": cost_var})
    with open("models_log.txt", "a") as file:
        file.write(str(models) + "\n")
    print(models)
    print(cost_var)


while cost_var > 1:
    gradient_descent()

# my_data["Year"] = round(my_data["Year"] + subtract_value)

x1_coordinate = my_data.Year[len(my_data) - 1]
y1_coordinate = line_function(x1_coordinate, w, b)
x2_coordinate = my_data.Year[0]
y2_coordinate = line_function(x2_coordinate, w, b)
print(w)
my_data.plot.scatter(x="Year", y="Population", ylabel="Population in Millions")

plt.plot([x1_coordinate, x2_coordinate], [y1_coordinate, y2_coordinate])
plt.show()
model(w, b, alpha, cost_var)
