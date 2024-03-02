from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

@app.get("/")
async def read_main(token: str = None):
    if token != "your_secret_token":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"message": "Hello, World!"}
