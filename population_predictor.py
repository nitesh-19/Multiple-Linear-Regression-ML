from linear_regression import LinearRegression

data_path = r".\data\Housing.csv"
feature_columns = [1, 2]
target_column = 0
trainer = LinearRegression(DATA_PATH=data_path, feature_columns_index=feature_columns,
                           target_column_index=target_column, ALPHA=0.8)

