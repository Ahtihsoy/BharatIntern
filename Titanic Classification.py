import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Step 1: Load the data
train_file = '/home/username/Desktop/titanic/train.csv'
test_file = '/home/username/Desktop/titanic/test.csv'
submission_file = '/home/username/Desktop/titanic/gender_submission.csv'

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
submission_data = pd.read_csv(submission_file)

# Step 2: Explore and preprocess the data
# Combine train and test data for preprocessing
combined_data = pd.concat([train_data.drop(columns=['Survived']), test_data], axis=0)

# Handle missing values
combined_data['Age'].fillna(combined_data['Age'].mean(), inplace=True)
combined_data['Fare'].fillna(combined_data['Fare'].mean(), inplace=True)
combined_data['Embarked'].fillna(combined_data['Embarked'].mode()[0], inplace=True)

# Encoding categorical variables
combined_data = pd.get_dummies(combined_data, columns=['Sex', 'Embarked'])

# Drop unnecessary columns
combined_data.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# Display the CSV table
display(combined_data.head())
import matplotlib.pyplot as plt

# Example: Bar graph of passenger count by Sex
sex_counts = combined_data['Sex_male'].value_counts()  # Assuming 'Sex_male' is the encoded column for males
sex_counts.plot(kind='bar', color=['blue', 'pink'])
plt.title('Passenger Count by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.xticks([0, 1], ['Female', 'Male'], rotation=0)
plt.show()
import matplotlib.pyplot as plt

# Example: Scatter plot of Age vs. Fare
plt.figure(figsize=(8, 6))
plt.scatter(combined_data['Age'], combined_data['Fare'], alpha=0.5, c='blue', edgecolors='none')
plt.title('Age vs. Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.grid(True)
plt.show()
# Example: Histogram of Age distribution
plt.figure(figsize=(8, 6))
plt.hist(combined_data['Age'], bins=20, color='green', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# Example: Box plot of Fare distribution by Pclass
plt.figure(figsize=(8, 6))
combined_data.boxplot(column='Fare', by='Pclass', grid=True)
plt.title('Fare Distribution by Pclass')
plt.suptitle('')  # Remove default title
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.show()
print(combined_data.head())
# Split back into train and test sets
X_train = combined_data.iloc[:len(train_data)]
X_test = combined_data.iloc[len(train_data):]

y_train = train_data['Survived']

# Step 3: Train a model
# Example: Using RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Make predictions
predictions = model.predict(X_test)

# Step 5: Evaluate the model (example: using train-test split for simplicity)
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model.fit(X_train_split, y_train_split)
val_predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)
print(f'Validation accuracy: {accuracy:.4f}')

