from linear_regression import LinearRegression

trainer = LinearRegression(ITERATION_SAMPLING_VALUE=5, ALPHA=0.0000005, ITERATIONS_LIMIT=391)
trainer.DATA_PATH = r".\data\Fish.csv"
trainer.x_index_list = [5]
trainer.y_index = 1
trainer.should_scale_data = False
trainer.w = [0.90497442]
trainer.b = [0.90497442]
trainer.run_trainer()

