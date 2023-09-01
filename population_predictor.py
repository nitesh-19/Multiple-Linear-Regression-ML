from linear_regression import LinearRegression

trainer = LinearRegression(ITERATION_SAMPLING_VALUE=5, ALPHA=0.005, ITERATIONS_LIMIT=100000)
trainer.DATA_PATH = r".\data\Fish.csv"
trainer.x_index_list = [5]
trainer.y_index = 1
trainer.should_scale_data = False

trainer.run_trainer()

