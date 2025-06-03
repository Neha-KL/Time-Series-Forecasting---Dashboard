import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv('TSLA.csv', parse_dates=['Date'], index_col='Date')
df = df[['Close']].dropna()

st.title("Tesla Stock Forecast Dashboard")

# Date range selector
start_date = st.date_input("Start date", df.index.min().date())
end_date = st.date_input("End date", df.index.max().date())

if start_date > end_date:
    st.error("Error: End date must fall after start date.")
else:
    # Filter data by selected date range
    filtered_df = df.loc[start_date:end_date]

    st.write(f"Data from {start_date} to {end_date}")
    
    if len(filtered_df) < 20:
        st.warning("Selected date range is too small for meaningful forecasting.")
    else:
        # Train-test split ratio slider
        split_ratio = st.slider("Train data ratio", min_value=0.5, max_value=0.95, value=0.8)

        split_index = int(len(filtered_df) * split_ratio)
        train = filtered_df['Close'][:split_index]
        test = filtered_df['Close'][split_index:]

        # ARIMA parameters input
        st.sidebar.header("ARIMA Model Parameters")
        p = st.sidebar.number_input("AR term (p)", min_value=0, max_value=10, value=5)
        d = st.sidebar.number_input("Difference term (d)", min_value=0, max_value=2, value=1)
        q = st.sidebar.number_input("MA term (q)", min_value=0, max_value=10, value=0)

        # Fit ARIMA model
        try:
            model = ARIMA(train, order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            test = test[:len(forecast)]

            # Plot results
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(test.index, test.values, label='Actual Prices', color='blue')
            ax.plot(test.index, forecast.values, label='Forecasted Prices', color='orange')
            ax.set_title(f'Tesla Stock Forecast vs Actual (ARIMA({p},{d},{q}))')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)

            # Show error metric
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(test, forecast)
            st.write(f"Mean Squared Error: {mse:.2f}")

        except Exception as e:
            st.error(f"Error in model fitting or forecasting: {e}")
