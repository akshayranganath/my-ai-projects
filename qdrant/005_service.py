from fastapi import FastAPI
from neural_search_with_city_filter_004 import NeuralSearcher

app = FastAPI()

neural_searcher = NeuralSearcher(collection_name="startups")

@app.get("/search")
def search_startup(q:str, city:str="Berlin"):
    return {
        "result": neural_searcher.search(q, city_of_interest=city)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)