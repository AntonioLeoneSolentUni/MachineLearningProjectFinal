import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from seaborn.external.appdirs import system
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go


try:
    data_cleaned = st.session_state.DataCleaned
    target_col = st.session_state.TargetCol
except Exception as e:
    st.warning("No Data found, Please provide a dataset.")
    st.stop()
with st.sidebar:
    st.write("Your Data: ", data_cleaned.head())


# Step 3: LSTM Model for Forecasting
st.subheader("AI Modeles for forecasting.")
st.write("Forecasting related to the Dataset given")
def LSTMRun():
    try:
        if st.session_state.mode == "Monthly Income":
            tab1, tab2, tab3 = st.tabs(["LST-Model", "ARIMA Model", "Facebook Prophet model"])
        else:
            tab1, tab2 = st.tabs(["LST-Model", "ARIMA Model"])

        # Scale the data
        scaler = MinMaxScaler()
        data_cleaned_scaled = scaler.fit_transform(data_cleaned[[target_col]])

        # Create sequences for LSTM
        sequence_length = 10
        X, y = [], []
        for i in range(len(data_cleaned_scaled) - sequence_length):
            X.append(data_cleaned_scaled[i:i + sequence_length])
            y.append(data_cleaned_scaled[i + sequence_length])

        X, y = np.array(X), np.array(y)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # Define LSTM model
        model = Sequential([
            LSTM(64, input_shape=(X.shape[1], X.shape[2]), activation='relu', return_sequences=False),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        st.write("Training LSTM model...")
        model.fit(X_train, y_train, epochs=200, batch_size=4, validation_data=(X_test, y_test), verbose=1)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_rescaled = scaler.inverse_transform(y_pred)
        y_test_rescaled = scaler.inverse_transform(y_test)
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)




        # Plot predictions vs actual
        figLSTM = go.Figure()
        figLSTM.add_trace(go.Scatter(y=y_test_rescaled.flatten(), mode='lines', name='Actual'))
        figLSTM.add_trace(go.Scatter(y=y_pred_rescaled.flatten(), mode='lines', name='Predicted'))
        figLSTM.update_layout(title="LSTM Predictions vs Actual", xaxis_title="Index", yaxis_title=target_col)

        #ARIMA model
        ArimaModel = ARIMA(data_cleaned[target_col],order=(1,2,2))
        modelFit = ArimaModel.fit()
        st.write(modelFit.summary())

        test = pd.array(data_cleaned[target_col]).tolist()

        test.append(modelFit.forecast(50))



        # Plot predictions vs actual
        figARIMA = go.Figure()
        figARIMA.add_trace(go.Scatter(y=data_cleaned[target_col], mode='lines', name='Actual'))
        figARIMA.add_trace(go.Scatter(y=test, mode='lines', name='Predicted'))
        figARIMA.update_layout(title="ARIMA Predictions vs Actual", xaxis_title="Index", yaxis_title=target_col)
        with tab1:
            st.plotly_chart(figLSTM)
            st.write("Loss LSTM: ", test_loss)
            st.write("LSTM mae: ", test_mae)
        with tab2:
            st.plotly_chart(figARIMA)
            st.write("ARIMA mae: ",modelFit.mae)

        if st.session_state.mode == "Monthly Income":
            # Facebook Prophet
            data_cleaned.reset_index(drop=True, inplace=True)
            currentDate = datetime.now()
            prophetDataframe = data_cleaned.loc[:, ['Month', target_col]]
            allMonthsValue = data_cleaned.loc[:, 'Month']

            prophetDataframe['Month'] = pd.to_datetime(prophetDataframe['Month'], format='%m', errors='coerce')

            # Iterate over the DataFrame rows
            for index in range(len(prophetDataframe)):
                # Subtract the number of months from the current date
                new_date = currentDate - relativedelta(months=allMonthsValue[index])

                # Update the 'Month' column with the new date formatted as 'YYYY-MM-DD'
                prophetDataframe.at[index, 'Month'] = new_date

            prophetDataframe.rename(columns={'Month': 'ds', target_col: 'y'}, inplace=True)
            st.write(prophetDataframe)
            facebookPropeht = Prophet()
            facebookPropeht.fit(prophetDataframe)
            future = facebookPropeht.make_future_dataframe(periods=20)


            forecastProphet = facebookPropeht.predict(future)

            fig = facebookPropeht.plot(forecastProphet)
            with tab3:
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Model training/prediction error: {e}")
        st.stop()

target_col = st.selectbox("Choose which data should be forecasted:", options = data_cleaned.columns)
st.write("Note that the more Data you store and feed the Forecasting Model the more precise it can get.")
st.warning("Therefore it is only a suggestion that will not reflect the real occurrence.")

st.button("Rerun LSTM model:", on_click= LSTMRun)



