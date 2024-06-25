
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

#model
data = {
    'Weather_Conditions': np.random.randint(1, 5, 100),
    'Road_Surface_Conditions': np.random.randint(1, 5, 100),
    'Light_Conditions': np.random.randint(1, 5, 100),
    'Number_of_Vehicles_Involved': np.random.randint(1, 5, 100),
    'Speed_Limit': np.random.randint(20, 120, 100),
    'Time_of_Day': np.random.randint(0, 24, 100),
    'Day_of_the_Week': np.random.randint(1, 8, 100),
    'Accident_Severity': np.random.randint(1, 4, 100),
    'Number_of_passengers' : np.random.randint(1,5,100)
}

df = pd.DataFrame(data)

# Define the dependent variable and independent variables
X = df.drop('Accident_Severity', axis=1)
y = df['Accident_Severity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model for future use
filename = 'accident_severity_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)


# Load the saved model
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Define a hypothetical set of independent variables
hypothetical_data = pd.DataFrame({
    'Weather_Conditions': [4],
    'Road_Surface_Conditions': [2],
    'Light_Conditions': [0],
    'Number_of_Vehicles_Involved': [5],
    'Speed_Limit': [120],
    'Time_of_Day': [14],
    'Day_of_the_Week': [5],
    'Number_of_passengers':[5]
})

# Predict accident severity
predicted_severity = loaded_model.predict(hypothetical_data)
print(f"Predicted Accident Severity: {predicted_severity[0]}")
