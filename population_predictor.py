from linear_regression import LinearRegression

trainer = LinearRegression()
trainer.DATA_PATH = r".\data\Housing.csv"
trainer.x_index_list = [1, 3, -1]
trainer.y_index = 0
print(trainer.get_training_data())
