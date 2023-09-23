from joblib import load
import pandas as pd

x_test = pd.read_csv('./test_processed_NA.csv', index_col=0)
print(x_test.head())
x = x_test.drop(['PatientID'], axis=1)


loaded_model = load('train_processed_NA_09_22.joblib')
predictions = loaded_model.predict(x)

print(predictions)

result = pd.DataFrame()

result['PatientID'] = x_test['PatientID']
result['HeartDisease'] = predictions

result.to_csv("test_processed_NA_0922_random_forest.csv",index=False)