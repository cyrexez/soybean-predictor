# predict.py
import pandas as pd
import pickle
import sys

def load_model():
    """Load the trained model and vectorizer"""
    with open('model.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    
    with open('vectorizer.pkl', 'rb') as f_vec:
        vectorizer = pickle.load(f_vec)
    
    return model, vectorizer

def preprocess_input(data_dict):
    """Preprocess input data to match training format"""
    # Convert cultivar to lowercase with underscores if present
    if 'Cultivar' in data_dict:
        data_dict['Cultivar'] = data_dict['Cultivar'].lower().replace(' ', '_')
    return data_dict

def predict(input_data):
    """
    Make prediction on new data
    
    Parameters:
    -----------
    input_data : dict or list of dicts
        Input features. Example:
        {
            'Season': 0,
            'Cultivar': 'neo_760_ce',
            'Repetition': 1,
            'PH': 58.80,
            'IFP': 15.20,
            'NLP': 98.21,
            'NPG': 77.80,
            'NPGL': 1.81,
            'NSM': 5.21,
            'HG': 52.20
        }
    
    Returns:
    --------
    predictions : float or list of floats
        Predicted grain yield (GY)
    """
    # Load model and vectorizer
    model, vectorizer = load_model()
    
    # Handle single dict or list of dicts
    if isinstance(input_data, dict):
        input_data = [input_data]
        single_input = True
    else:
        single_input = False
    
    # Preprocess
    input_data = [preprocess_input(d) for d in input_data]
    
    # Vectorize
    X = vectorizer.transform(input_data)
    
    # Predict
    predictions = model.predict(X)
    
    # Return single value or list
    return predictions[0] if single_input else predictions.tolist()

if __name__ == "__main__":
    # Example usage
    sample_input = {
        'Season': 0,
        'Cultivar': 'neo_760_ce',
        'Repetition': 1,
        'PH': 58.80,
        'IFP': 15.20,
        'NLP': 98.21,
        'NPG': 77.80,
        'NPGL': 1.81,
        'NSM': 5.21,
        'HG': 52.20
    }
    
    print("Making prediction...")
    prediction = predict(sample_input)
    print(f"Predicted Grain Yield (GY): {prediction:.2f}")
    
    # Multiple predictions
    multiple_inputs = [
        sample_input,
        {
            'Season': 1,
            'Cultivar': 'manu_ipro',
            'Repetition': 1,
            'PH': 81.20,
            'IFP': 18.00,
            'NLP': 98.80,
            'NPG': 73.00,
            'NPGL': 1.75,
            'NSM': 7.40,
            'HG': 45.59
        }
    ]
    
    predictions = predict(multiple_inputs)
    print(f"\nMultiple predictions: {predictions}")