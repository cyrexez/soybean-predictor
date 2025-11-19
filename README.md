# Description of the Project
This project is a soybean cultivar predictor inspired by previous research in the field.
It was trained using an automated loop that evaluates multiple regression models and selects the one with the highest R² score.
When executed, the project exports the final model using Pickle.
The predict.py file runs automatically to test the algorithm, while test.py launches a FastAPI server for making remote predictions.
All components can be run using uv, and the project includes a Dockerfile for containerized deployment.

# Dataset
https://github.com/brunobro/dataset-forty-soybean-cultivars-from-subsequent-harvests

# Research paper
https://editorapantanal.com.br/journal/index.php/taes/en/article/view/8

# Instructions on how to run the project
docker build -t soybean-predictor .


docker run -p 8000:8000 soybean-predictor

# Website

First create a repo on github
# Initialize git (if not already)
git init

git add .

git commit -m "Initial commit"

# Create a repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/soybean-predictor.git

git push -u origin main

# Deploy on Render:

Go to https://render.com

Sign up (free)

Click "New +" → "Web Service"

Connect your GitHub repo

Configure:
  Name: soybean-predictor
  Environment: Docker
  Plan: Free (or paid for better performance)
  
Click "Create Web Service"

Render will automatically detect the Dockerfile

Render will proceed to build and deploy automatically

Give you a URL like: https://soybean-predictor.onrender.com

Website: https://soybean-predictor.onrender.com

Note: It was deployed using render. However, there seems to be a lag in making GET requests. It does work.


