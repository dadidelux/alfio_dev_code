import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# from arima_forecast import get_parcel_sum_daily , 

from fastapi import FastAPI, Query
import mysql.connector
from mysql.connector import Error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
# Define the path to the alfio_dev folder
# deployment
alfio_dev_path = "/opt/render/project/src/"
#localhost
# alfio_dev_path = "../alfio_dev_p/"

# Construct the path to the CSV file
csv_file_path = os.path.join(alfio_dev_path, "data", "mabuhay_price.csv")
output_file_path = os.path.join(alfio_dev_path, "pkl_output", "mabuhay_price.pkl")

app = FastAPI()

origins = ["https://mabuhaypadala.online/","https://212fa74c-1e7d-4f93-ae4c-8d14a89712dc-00-3j2c0tris6tsn.spock.replit.dev","*"]

# Add CORS middleware to allow connections from the specified origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  
    allow_headers=["*"],  
)

# Load the CSV file
df = pd.read_csv(csv_file_path)

@app.get("/create-index")
def create_faiss_index():
    titles = df["mergedata"].tolist()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(titles, convert_to_tensor=True)
    embeddings_np = embeddings.cpu().detach().numpy()

    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    faiss.write_index(index, output_file_path)
    return {"message": f"FAISS index created and saved to {output_file_path}"}

@app.get("/search")
def search_similar_titles(query_title: str, top_k: int = 5):
    if not os.path.exists(output_file_path):
        raise HTTPException(status_code=404, detail="FAISS index not found. Please create the index first.")

    index = faiss.read_index(output_file_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query_title], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().detach().numpy()

    distances, indices = index.search(query_embedding_np, top_k)
    
    # Convert numpy.float32 to Python float for JSON serialization
    similarities = [float(sim) for sim in (1 - distances[0])]

    similar_titles = df.iloc[indices[0]]["mergedata"].tolist()
    shipping_prices = df.iloc[indices[0]]["shippingfee"].tolist()

    results = [{"title": title, "shipping_price": price, "similarity": dist}
            for title, price, dist in zip(similar_titles, shipping_prices, similarities)]
    return results

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
    logger.debug("Processing forecast")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    daily_df = df.resample("D").ffill()
    smoothed_df = daily_df.rolling(window=7, min_periods=1).mean()
    train_size = int(len(smoothed_df) * 0.8)
    train, test = smoothed_df.iloc[:train_size], smoothed_df.iloc[train_size:]
    logger.debug("Training ARIMA")
    arima_model = ARIMA(train, order=(5, 1, 1))
    arima_results = arima_model.fit()
    forecast = arima_results.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    forecast_mean = forecast_mean.apply(lambda x: str(x) if np.isfinite(x) else None)
    logger.debug("Convert the forecast index (dates) to string format")
    forecast_mean.index = forecast_mean.index.strftime("%Y-%m-%d")

    mae = mean_absolute_error(test, forecast_mean.astype(float))
    mse = mean_squared_error(test, forecast_mean.astype(float))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test - forecast_mean.astype(float)) / test)) * 100

    mae = mae if np.isfinite(mae) else None
    mse = mse if np.isfinite(mse) else None
    rmse = rmse if np.isfinite(rmse) else None
    mape = mape if np.isfinite(mape) else None
    logger.debug("Returning Result")
    return {
        "dataset": df.reset_index().to_dict(orient="records"),
        "forecast": forecast_mean.to_dict(),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        # "mape": mape,
    }


@app.get("/arima_forecast")
def arima_fast_api(startdate: str = Query(None), enddate: str = Query(None)):
    data = get_parcel_sum_daily(startdate, enddate)
    if data.empty:
        return {"error": "No data found for the given date range"}
    forecast_results = arima_forecast(data)
    return forecast_results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
