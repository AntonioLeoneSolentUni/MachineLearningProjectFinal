import streamlit as st

try:
    data_cleaned = st.session_state.DataCleaned
except Exception as e:
    st.warning("No Data found, Please provide a dataset.")
    st.stop()
with st.sidebar:
    st.write("Your Data: ", data_cleaned.head())

data_cleaned = st.session_state.data
target_col = st.session_state.TargetCol




st.subheader("Step 4: Decision-Making Support")
st.write("---")
st.write("This section helps you with decision making regarding to your savings and spending")
st.write("You can select what your savings target is and what you saved up so far.")
st.write("It is real time, this means you can adjust the parameter below and see how it affects your saving goal.")
try:
    current_savings = data_cleaned[target_col].iloc[-1]  # Ensure data_cleaned is defined before this step
    interval = st.selectbox("Select Interval:", ["Daily", "Weekly", "Monthly"])
    savings_goal = st.number_input("Set Your Savings Target (£):", min_value=0.0, step=100.0)

    if current_savings < savings_goal:
        st.warning(f"You need to save an additional £{savings_goal - current_savings:.2f} to meet your target.")
    else:
        st.success("Congratulations! You have met your savings goal.")
except Exception as e:
    st.error(f"Decision-making support error: {e}")

# Step 5: Interactivity and Real-Time Updates
st.subheader("Step 5: Interactivity and Real-Time Updates")
try:
    real_time_savings = st.slider("Adjust Current Savings (£):", min_value=0, max_value=int(current_savings + 5000), value=int(current_savings))
    updated_goal_status = f"Met Goal status of {savings_goal}" if real_time_savings >= savings_goal else (f"Not Met it needs: { savings_goal - real_time_savings  } £ to reach Goal.")
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
