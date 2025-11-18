# Description of the Project
This project is a soybean cultiuvator predictor based on the work of 
The project was trained using the best regression model using a loop and picking the model with the highest R2 model.
The project when run exports using pickle.
The Predict python file is automatically run as a way to test the algorithm.
The test.py is used to run a FastAPI that can be used to predict remotely.
Everyhing is run with uv and Dockerfile

# Dataset
https://github.com/brunobro/dataset-forty-soybean-cultivars-from-subsequent-harvests

# Research paper
https://editorapantanal.com.br/journal/index.php/taes/en/article/view/8

# Instructions on how to run the project
docker build -t soybean-predictor . /n
docker run -p 8000:8000 soybean-predictor

