from joblib import load
import pandas as pd

x_test = pd.read_csv('./test_processed_dropped_features.csv', index_col=0)
print(x_test.head())
x = x_test.drop(['PatientID'], axis=1)


loaded_model = load('project1_0921_random_forest.joblib')
predictions = loaded_model.predict(x)

print(predictions)

result = pd.DataFrame()

result['PatientID'] = x_test['PatientID']
result['HeartDisease'] = predictions

result.to_csv("project1_result_0922_random_forest.csv",index=False)