import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

try:
    data_cleaned = st.session_state.DataCleaned
except Exception as e:
    st.warning("No Data found, Please provide a dataset.")
    st.stop()
with st.sidebar:
    st.write("Your Data: ", data_cleaned.head())

data_cleaned = st.session_state.DataCleaned
target_col = st.session_state.TargetCol


# Select features and target
features = data_cleaned[
    [col for col in data_cleaned.columns if col != 'Savings for Property (£)']
]
target = data_cleaned['Savings for Property (£)']

# Normalize data
scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
featuresScaled = scalerX.fit_transform(features)
targetScaled = scalerY.fit_transform(target.values.reshape(-1, 1))

# Reshape data for LSTM input
X = featuresScaled.reshape(featuresScaled.shape[0], 1, featuresScaled.shape[1])
y = targetScaled

# Split into training and testing datasets
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM Model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), activation='relu', return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Linear activation for regression
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
model.fit(XTrain, yTrain, epochs=50, batch_size=4, validation_data=(XTest, yTest), verbose=1)

# Make Predictions
predictions = model.predict(XTest)
predictedValues = scalerY.inverse_transform(predictions)
predictedAverage = predictedValues.mean()



st.subheader("Step 4: Decision-Making Support")
st.write("---")
st.write("This section helps you with decision making regarding to your savings and spending")
st.write("You can select what your savings target is and what you saved up so far.")
st.write("It is real time, this means you can adjust the parameter below and see how it affects your saving goal.")
try:
    currentSavings = data_cleaned[target_col].iloc[-1]  # Ensure data_cleaned is defined before this step
    interval = st.selectbox("Select Interval:", ["Daily", "Weekly", "Monthly"])
    savingsGoal = st.number_input("Set Your Savings Target (£):", min_value=0.0, step=100.0)

    if interval == "Daily":
        intervalValue = 30
    if interval == "Weekly":
        intervalValue = 4
    if interval == "Monthly":
        intervalValue = 1

    savingsPerInterval = currentSavings / intervalValue
    requiredSavingsPerInterval = savingsGoal / intervalValue
    predictedSavingPerInterval =  predictedAverage / intervalValue
    st.write(f"to reach your goal, you need to save {requiredSavingsPerInterval:.2f} savings per interval.")
    if currentSavings < savingsGoal:
        st.warning(f"You need to save an additional £{savingsGoal - currentSavings:.2f} to meet your target.")
        st.warning(f"you need to save {savingsPerInterval:.2f} on a {interval} basis.")
        st.warning(f"Predicting saving for your saving goal is {predictedSavingPerInterval:.2f} on a {interval} basis.")
    else:
        st.success("Congratulations! You have met your savings goal.")
except Exception as e:
    st.error(f"Decision-making support error: {e}")

# Step 5: Interactivity and Real-Time Updates
st.subheader("Step 5: Interactivity and Real-Time Updates")
try:
    real_time_savings = st.slider("Adjust Current Savings (£):", min_value=0, max_value=int(currentSavings + 5000), value=int(currentSavings))
    updated_goal_status = f"Met Goal status of {savingsGoal}" if real_time_savings >= savingsGoal else (f"Not Met it needs: {savingsGoal - real_time_savings  } £ to reach Goal.")
    st.write(f"Updated Goal Status: {updated_goal_status}")
except Exception as e:
    st.error(f"Real-time updates error: {e}")

# Step 6: Scenario Planning and Forecasting
st.subheader("Step 6: Scenario Planning and Forecasting")
try:
    scenario_increase = st.number_input("Increase Savings by (%):", min_value=0, max_value=100, step=5)
    forecasted_savings = real_time_savings * (1 + scenario_increase / 100)
    st.write(f"If you increase savings by {scenario_increase}%, your forecasted savings will be £{forecasted_savings:.2f}.")
except Exception as e:
    st.error(f"Scenario planning error: {e}")
