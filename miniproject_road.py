import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
try:
    df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    print("Columns in DataFrame:", df.columns)
    print(df.head())  # Print the first few rows to inspect the data
except PermissionError:
    print("Permission denied: Unable to read the file. Please check if it is open or your permissions.")
    exit()

# Data Preprocessing
df.dropna(subset=['Age', 'Fare'], inplace=True)  # Drop rows with missing 'Age' or 'Fare' values
df.columns = df.columns.str.strip()  # Strip spaces from column names

# Exploratory Data Analysis
# Example: Survival Rate by Age
plt.figure(figsize=(10, 5))
sns.histplot(df, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('Survival Rate by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Example: Survival Rate by Passenger Class
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# Example: Survival Rate by Gender
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Example: Correlation Heatmap
# Select only numeric columns for the heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
