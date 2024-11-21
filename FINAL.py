# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/Users/harshit07/Desktop/MINI FINAL/updateddataset.csv')

# Convert DateTime column to pandas datetime object and extract features
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['year'] = df['DateTime'].dt.year
df['month'] = df['DateTime'].dt.month
df['day'] = df['DateTime'].dt.day
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek  # Monday=0, Sunday=6

# Set a seaborn style for consistent visualizations
sns.set(style="whitegrid")

# Visualization
# 1. Traffic flow across different junctions over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='DateTime', y='Vehicles', hue='Junction', data=df)
plt.title('Traffic Flow Across Different Junctions Over Time')
plt.xticks(rotation=45)
plt.show()

# 2. Pairplot of different features
sns.pairplot(df[['Junction', 'Vehicles', 'year', 'month', 'day', 'hour']])
plt.suptitle('Pairplot of Different Features', y=1.02)
plt.show()

# 3. Histogram of month-wise traffic flow
plt.figure(figsize=(10, 6))
sns.histplot(df, x='month', hue='Junction', multiple='stack', kde=False, bins=12)
plt.title('Month-wise Traffic Flow Across Junctions')
plt.xlabel('Month')
plt.ylabel('Number of Vehicles')
plt.show()

# 4. Lineplot for vehicles across the year
plt.figure(figsize=(10, 6))
sns.lineplot(x='year', y='Vehicles', hue='Junction', data=df)
plt.title('Number of Vehicles Across the Year for Different Junctions')
plt.show()

# Lineplot for vehicles across the month
plt.figure(figsize=(10, 6))
sns.lineplot(x='month', y='Vehicles', hue='Junction', data=df)
plt.title('Number of Vehicles Across the Month for Different Junctions')
plt.show()

# Lineplot for vehicles across the day
plt.figure(figsize=(10, 6))
sns.lineplot(x='day', y='Vehicles', hue='Junction', data=df)
plt.title('Number of Vehicles Across the Day for Different Junctions')
plt.show()

# Lineplot for vehicles across the hour
plt.figure(figsize=(10, 6))
sns.lineplot(x='hour', y='Vehicles', hue='Junction', data=df)
plt.title('Number of Vehicles Across the Hour for Different Junctions')
plt.show()

# 5. Countplot of traffic across different years
plt.figure(figsize=(10, 6))
sns.countplot(x='year', hue='Junction', data=df)
plt.title('Traffic Count Across Different Years')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

# 6. Plotting the number of vehicles across every day of the week for each junction
plt.figure(figsize=(12, 6))
sns.lineplot(x='day_of_week', y='Vehicles', hue='Junction', data=df)
plt.title('Number of Vehicles Across Each Day of the Week for Each Junction')
plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
plt.ylabel('Number of Vehicles')
plt.show()

# RMSE calculation function
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Regression Models for Predicting Vehicles
# Step 1: Load the dataset
df = pd.read_csv('/Users/harshit07/Desktop/MINI FINAL/updateddataset.csv')

# Step 2: Feature Engineering - Convert DateTime column
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['day'] = df['DateTime'].dt.day
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek

# Step 3: Get unique junctions
junctions = df['Junction'].unique()

# Step 4: Initialize results storage
results = []

# Step 5: Loop over each junction and apply models individually
for junction in junctions:
    print(f"Processing Junction {junction}...")
    
    # Filter data for the specific junction
    junction_df = df[df['Junction'] == junction]
    
    # Define Features and Target
    X = junction_df[['day', 'hour', 'day_of_week']]
    y = junction_df['Vehicles']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize a dictionary to store the results for this junction
    junction_results = {'Junction': junction}
    
    # Train models and calculate metrics
    ## Decision Tree Regressor
    dt_regressor = DecisionTreeRegressor(random_state=42)
    dt_regressor.fit(X_train, y_train)
    y_train_pred_dt = dt_regressor.predict(X_train)
    y_test_pred_dt = dt_regressor.predict(X_test)
    junction_results['DT_Train_RMSE'] = calculate_rmse(y_train, y_train_pred_dt)
    junction_results['DT_Test_RMSE'] = calculate_rmse(y_test, y_test_pred_dt)
    
    ## Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_train_pred_lr = lr_model.predict(X_train)
    y_test_pred_lr = lr_model.predict(X_test)
    junction_results['LR_Train_RMSE'] = calculate_rmse(y_train, y_train_pred_lr)
    junction_results['LR_Test_RMSE'] = calculate_rmse(y_test, y_test_pred_lr)
    
    ## Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    y_train_pred_rf = rf_regressor.predict(X_train)
    y_test_pred_rf = rf_regressor.predict(X_test)
    junction_results['RF_Train_RMSE'] = calculate_rmse(y_train, y_train_pred_rf)
    junction_results['RF_Test_RMSE'] = calculate_rmse(y_test, y_test_pred_rf)
    
    # Append the junction results to the results list
    results.append(junction_results)

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('/Users/harshit07/Desktop/MINI FINAL/results.csv', index=False)

# Classification Model for Predicting Traffic Level
data = pd.read_csv('/Users/harshit07/Desktop/MINI FINAL/updateddataset.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['Hour'] = data['DateTime'].dt.hour
data['DayOfWeek'] = data['DateTime'].dt.dayofweek
data['Month'] = data['DateTime'].dt.month

# Define features and target
features = ['Hour', 'DayOfWeek', 'Month', 'Junction']
X_classification = data[features]
y_classification = data['Traffic_Level']

# Split the data into training and test sets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_class, y_train_class)

# Evaluate the model
y_pred_class = rf_classifier.predict(X_test_class)
print("Classification Accuracy Score:", accuracy_score(y_test_class, y_pred_class))
print("Classification Report:\n", classification_report(y_test_class, y_pred_class))

# Dynamic prediction function
def get_user_input_and_predict():
    datetime_str = input("Enter the Date and Time (YYYY-MM-DD HH:MM:SS): ")
    junction = int(input("Enter the Junction ID: "))
    
    # Process the input
    dt = pd.to_datetime(datetime_str)
    input_data = pd.DataFrame({
        'Hour': [dt.hour],
        'DayOfWeek': [dt.dayofweek],
        'Month': [dt.month],
        'Junction': [junction]
    })
    
    # Make the prediction
    prediction = rf_classifier.predict(input_data)
    print(f"Predicted Traffic Level: {prediction[0]}")

# Run the dynamic prediction function
get_user_input_and_predict()
