import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Streamlit App Title
st.title("📈 Stock Volatility & GARCH(1,1) Forecast App")

# User Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA)", value="TSLA").upper()
start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2022-10-14"))

# Run analysis when button is clicked
if st.button("Run Analysis"):

    try:
        # Download historical data
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            st.warning("No data found for the given ticker and date range.")
        else:
            # Calculate daily returns in percentage
            data['Daily_Returns'] = 100 * data['Close'].pct_change()
            data.dropna(inplace=True)

            # Plot Daily Returns
            st.subheader("📊 Daily Returns Plot")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data['Daily_Returns'], label='Daily Returns')
            ax.set_title(f'Daily Returns of {ticker}')
            ax.set_ylabel("Return (%)")
            ax.set_xlabel("Date")
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend()
            st.pyplot(fig)

            # Volatility Calculations using numpy instead of math
            daily_vol = data['Daily_Returns'].std()
            monthly_vol = np.sqrt(21) * daily_vol
            annual_vol = np.sqrt(252) * daily_vol

            # Display Volatility
            st.subheader("📉 Volatility Summary")
            vol_df = pd.DataFrame([{
                'Daily Volatility %': round(daily_vol, 2),
                'Monthly Volatility %': round(monthly_vol, 2),
                'Annual Volatility %': round(annual_vol, 2)
            }], index=[ticker])
            st.dataframe(vol_df)

            # Fit GARCH(1,1) Model
            st.subheader("📐 GARCH(1,1) Model Parameters")
            garch = arch_model(data['Daily_Returns'], p=1, q=1, mean='constant', vol='GARCH', dist='normal')
            result = garch.fit(disp='off')
            st.write(result.params)

            # Forecast Variance for Next 5 Days
            forecast = result.forecast(horizon=5)
            variance_forecast = forecast.variance.tail(1)
            variance_forecast = variance_forecast.values.reshape(-1, 5)
            variance_df = pd.DataFrame(variance_forecast, columns=[f"Day {i+1}" for i in range(5)])
            st.subheader("📊 5-Day Ahead Variance Forecast")
            st.dataframe(variance_df)

            # Convert variance to volatility (standard deviation)
            st.subheader("🔁 5-Day Ahead Volatility Forecast (Standard Deviation %)")
            volatility_df = np.sqrt(variance_df)
            st.dataframe(volatility_df)

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")
