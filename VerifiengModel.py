import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv('Monthly_Income_OutgoingsV1.csv')
target_col = "Gas Bill (£)"


# Step 3: LSTM Model for Forecasting
# Scale the data
scaler = MinMaxScaler()
data_cleaned_scaled = scaler.fit_transform(df[[target_col]])

        # Create sequences for LSTM
sequence_length = 10
X, y = [], []
for i in range(len(data_cleaned_scaled) - sequence_length):
    X.append(data_cleaned_scaled[i:i + sequence_length])
    y.append(data_cleaned_scaled[i + sequence_length])

X, y = np.array(X), np.array(y)

X = X[:,:,0]

# Step 3: Encode categorical data
label_encoder = LabelEncoder()
label_encoders = {}


# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
rf_pred = rf_model.predict(X_test_scaled)
print(f"\nRandom Forest Accuracy: {accuracy_score(y_test, rf_pred)}")  # Expected: Accuracy scor

# Linear Regression (for continuous outcomes, could be cancer progression score or survival time)
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)  # Here y_train would be continuous in this case

# Predict and Evaluate
linear_pred = linear_model.predict(X_test_scaled)
print(f"\nLinear Regression R^2: {r2_score(y_test, linear_pred)}")  # Expected: R^2 score

# Support Vector Classifier (SVC)
svm_model = SVC(kernel='linear', random_state=42)  # Use 'linear' kernel
svm_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
svm_pred = svm_model.predict(X_test_scaled)
print(f"\nSVM Accuracy: {accuracy_score(y_test, svm_pred)}")  # Expected: Accuracy score

# Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
nb_pred = nb_model.predict(X_test_scaled)
print(f"\nNaive Bayes Accuracy: {accuracy_score(y_test, nb_pred)}")  # Expected: Accuracy score
