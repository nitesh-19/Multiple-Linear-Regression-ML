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
my_data["Population"] = round(my_data["Population"] * 10 ** -6)  # Converting population to in Millions
print(my_data)

# def cost_function():
cost_list = []
m = len(my_data)


def line_function(x, w, b):
    y = w * x + b
    return y


def cost_function(w, b):
    summation = 0
    for i in range(0, m):
        y_cap = line_function(my_data.Year[i], w, b)
        summation += (y_cap - my_data.Population[i]) ** 2
    cost = summation / (2 * m)
    return cost


def model():
    pass


print(cost_function(1, 0))
my_data.plot.scatter(x="Year", y="Population", ylabel="Population in Millions")
# plt.plot([558, 1323])
plt.show()
