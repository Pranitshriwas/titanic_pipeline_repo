import pandas as pd

df = pd.read_csv('data/tested.csv')

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

df.to_csv('data/processed_data.csv', index=False)
print(" Data transformed and saved to 'data/processed_data.csv'")

print(df.head())