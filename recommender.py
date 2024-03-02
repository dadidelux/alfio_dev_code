import os
from fastapi import FastAPI, HTTPException
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

import os
# Define the path to the alfio_dev folder
alfio_dev_path = "/opt/render/project/src/"

# Construct the path to the CSV file
csv_file_path = os.path.join(alfio_dev_path, "data", "mabuhay_price.csv")
output_file_path = os.path.join(alfio_dev_path, "pkl_output", "mabuhay_price.pkl")

app = FastAPI()


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
