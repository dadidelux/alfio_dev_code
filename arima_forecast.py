from fastapi import FastAPI, Query
import mysql.connector
from mysql.connector import Error
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

app = FastAPI()

DB_CONFIG = {
    "host": "srv946.hstgr.io",
    "database": "u955224677_mpc_prod",
    "user": "u955224677_mpc_prod",
    "password": "s:A6*O|BWI4",
}


def get_parcel_sum_daily(start_date, end_date):
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            query = "SELECT * FROM `parcel_sum_daily` WHERE `date` BETWEEN %s AND %s"
            cursor.execute(query, (start_date, end_date))
            records = cursor.fetchall()
            return pd.DataFrame(records)
    except Error as e:
        return pd.DataFrame()
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


def arima_forecast(df):
    print("processing forecast")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    daily_df = df.resample("D").ffill()
    smoothed_df = daily_df.rolling(window=7, min_periods=1).mean()
    train_size = int(len(smoothed_df) * 0.8)
    train, test = smoothed_df.iloc[:train_size], smoothed_df.iloc[train_size:]
    print("Training ARIMA")
    arima_model = ARIMA(train, order=(3, 1, 1))
    arima_results = arima_model.fit()
    forecast = arima_results.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    forecast_mean = forecast_mean.apply(lambda x: str(x) if np.isfinite(x) else None)
    print("Convert the forecast index (dates) to string format")
    # Convert the forecast index (dates) to string format
    forecast_mean.index = forecast_mean.index.strftime("%Y-%m-%d")

    mae = mean_absolute_error(test, forecast_mean.astype(float))
    mse = mean_squared_error(test, forecast_mean.astype(float))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test - forecast_mean.astype(float)) / test)) * 100

    # Handle nan values in evaluation metrics
    mae = mae if np.isfinite(mae) else None
    mse = mse if np.isfinite(mse) else None
    rmse = rmse if np.isfinite(rmse) else None
    mape = mape if np.isfinite(mape) else None
    print("Returning Result")
    return {
        "dataset": df.reset_index().to_dict(orient="records"),
        "forecast": forecast_mean.to_dict(),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
    }


@app.get("/arima_forecast")
def arima_fast_api(startdate: str = Query(None), enddate: str = Query(None)):
    data = get_parcel_sum_daily(startdate, enddate)
    if data.empty:
        return {"error": "No data found for the given date range"}
    forecast_results = arima_forecast(data)
    print(forecast_results)
    return forecast_results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
