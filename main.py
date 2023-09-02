from linear_regression import LinearRegression, predict_with_model

data_path = r".\data\insurance.csv"  # Dataset having features and target in a vertical arrangement.
feature_columns = [0, 2, 3, 4]  # Index of the columns in the dataset to be fed as features to the trainer
target_column = 6  # Index of the column in the dataset to be fed as the target to the trainer.


### TRAIN MODEL ###
# Create trainer object which will save the model to a text file after performing Multiple Linear Regression.
trainer = LinearRegression(DATA_PATH=data_path, feature_columns_index=feature_columns,
                           target_column_index=target_column, ALPHA=0.5, create_test_set=False, ITERATIONS_LIMIT=10000)


### PREDICT FROM THE MODEL ###
parameters = [53, 33.25, 0, 0]  # Features from the test dataset needed to predict the target.
target_prediction = predict_with_model(
    parameters)  # Predict the target by extracting the weights from the last saved model.
print(f"Prediction for the input features {parameters} is {target_prediction}")
