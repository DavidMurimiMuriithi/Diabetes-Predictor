import pandas as pd

df = pd.read_csv('diabetes.csv')

# print(df.head())


new_df = df['BloodPressure']
max_gl = df['BloodPressure'].max()
min_gl = df['BloodPressure'].min()

print(new_df)
# print(max_gl)
# print(min_gl)