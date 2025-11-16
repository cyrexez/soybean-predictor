# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import uvicorn
from typing import List, Optional

# Initialize FastAPI app
app = FastAPI(
    title="Soybean Yield Predictor API",
    description="API for predicting soybean grain yield based on cultivar characteristics",
    version="1.0.0"
)

# Load model and vectorizer at startup
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("✓ Model and vectorizer loaded successfully")
except FileNotFoundError:
    print("ERROR: Model files not found. Please run train.py first!")
    model = None
    vectorizer = None

# Define input schema
class SoybeanInput(BaseModel):
    Season: int = Field(..., description="Growing season (0, 1, 2, etc.)")
    Cultivar: str = Field(..., description="Cultivar name (e.g., 'NEO 760 CE' or 'neo_760_ce')")
    Repetition: int = Field(..., description="Experimental repetition number")
    PH: float = Field(..., description="Plant height", gt=0)
    IFP: float = Field(..., description="Insertion of first pod", gt=0)
    NLP: float = Field(..., description="Number of lateral pods", gt=0)
    NPG: float = Field(..., description="Number of pods with grains", gt=0)
    NPGL: float = Field(..., description="Number of pods per grain location", gt=0)
    NSM: float = Field(..., description="Number of seeds per meter", gt=0)
    HG: float = Field(..., description="Hundred grain weight", gt=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "Season": 0,
                "Cultivar": "NEO 760 CE",
                "Repetition": 1,
                "PH": 58.80,
                "IFP": 15.20,
                "NLP": 98.21,
                "NPG": 77.80,
                "NPGL": 1.81,
                "NSM": 5.21,
                "HG": 52.20
            }
        }

class PredictionResponse(BaseModel):
    predicted_grain_yield: float
    input_data: dict

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    count: int

def preprocess_input(data: dict) -> dict:
    """Preprocess input data to match training format"""
    data_copy = data.copy()
    if 'Cultivar' in data_copy:
        data_copy['Cultivar'] = data_copy['Cultivar'].lower().replace(' ', '_')
    return data_copy

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Soybean Yield Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict - Single prediction",
            "predict_batch": "/predict/batch - Batch predictions",
            "health": "/health - Health check",
            "docs": "/docs - Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "vectorizer_loaded": True
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: SoybeanInput):
    """
    Make a single prediction for soybean grain yield
    
    Returns the predicted grain yield (GY) based on input features
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please run train.py first.")
    
    try:
        # Convert to dict and preprocess
        data_dict = input_data.dict()
        processed_data = preprocess_input(data_dict)
        
        # Vectorize
        X = vectorizer.transform([processed_data])
        
        # Predict
        prediction = model.predict(X)[0]
        
        return {
            "predicted_grain_yield": round(float(prediction), 2),
            "input_data": data_dict
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(inputs: List[SoybeanInput]):
    """
    Make batch predictions for multiple soybean samples
    
    Accepts a list of input data and returns predictions for all
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please run train.py first.")
    
    if len(inputs) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 predictions per batch")
    
    try:
        # Convert to dicts and preprocess
        data_dicts = [input_data.dict() for input_data in inputs]
        processed_data = [preprocess_input(d) for d in data_dicts]
        
        # Vectorize
        X = vectorizer.transform(processed_data)
        
        # Predict
        predictions = model.predict(X)
        
        return {
            "predictions": [round(float(p), 2) for p in predictions],
            "count": len(predictions)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=8000)