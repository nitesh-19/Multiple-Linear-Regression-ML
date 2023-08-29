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
my_data["Population"] = round(my_data["Population"] * 10 ** -8)  # Converting population to in Millions
my_data["Year"] = round(my_data["Year"] + 30 - 2000)  # Converting population to in Millions

print(my_data)

# def cost_function():
cost_var = 1000000000000
m = len(my_data)
w = 0
b = 0
alpha =0.001


def line_function(x, w, b):
    y = w * x + b
    return y


def cost_function(w, b):
    global cost_var
    summation = 0
    for i in range(0, m):
        y_cap = line_function(my_data.Year[i], w, b)
        summation += (y_cap - my_data.Population[i]) ** 2
    cost = summation / (2 * m)
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
    for i in range(0, m):
        y_cap = line_function(my_data.Year[i], w, b)
        summation_b += (y_cap - my_data.Population[i])
    summation_w /= m
    summation_b /= m
    w_temp = w - alpha * summation_w
    b_temp = b - alpha * summation_b
    w = w_temp
    b = b_temp
    cost_function(w, b)


def model():
    pass


while cost_var > 0.1:
    gradient_descent()

x1_coordinate = my_data.Year[len(my_data) - 1]
y1_coordinate = line_function(x1_coordinate, w, b)
x2_coordinate = my_data.Year[0]
y2_coordinate = line_function(x2_coordinate, w, b)
print(w)
my_data.plot.scatter(x="Year", y="Population", ylabel="Population in Millions")

plt.plot([x1_coordinate, x2_coordinate], [y1_coordinate, y2_coordinate])
plt.show()
