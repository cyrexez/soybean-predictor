# train.py - Improved version with hyperparameter tuning
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Load data
print("Loading data...")
df = pd.read_csv("https://raw.githubusercontent.com/brunobro/dataset-forty-soybean-cultivars-from-subsequent-harvests/main/data.csv")

print(f"Dataset shape: {df.shape}")
print(f"Target variable (GY) - Mean: {df['GY'].mean():.2f}, Std: {df['GY'].std():.2f}")

# Preprocess: convert cultivar to lowercase with underscores
df['Cultivar'] = df['Cultivar'].str.lower().str.replace(r'\s+', '_', regex=True)

# Prepare X and y
X = df.drop('GY', axis=1)
y = df['GY']

# Convert to dictionaries and vectorize
X_dicts = X.to_dict('records')
dv = DictVectorizer(sparse=False)
X_vectorized = dv.fit_transform(X_dicts)

print(f"Number of features after vectorization: {X_vectorized.shape[1]}")

# Split data: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Hyperparameter tuning with GridSearchCV
print("\nPerforming hyperparameter tuning...")
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
}

ridge_base = Ridge(random_state=42)
grid_search = GridSearchCV(
    ridge_base, 
    param_grid, 
    cv=5, 
    scoring='r2',
    n_jobs=-1
)

# Combine train and val for cross-validation
X_train_val = np.vstack([X_train, X_val])
y_train_val = np.concatenate([y_train, y_val])

grid_search.fit(X_train_val, y_train_val)

print(f"Best alpha: {grid_search.best_params_['alpha']}")
print(f"Best CV R² score: {grid_search.best_score_:.4f}")

# Train final model with best parameters
best_model = grid_search.best_estimator_

# Evaluate on test set
y_test_pred = best_model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

# Also evaluate on validation set for comparison
y_val_pred = best_model.predict(X_val)
val_r2 = r2_score(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)

print("\n" + "=" * 50)
print("RIDGE REGRESSION MODEL TRAINING RESULTS")
print("=" * 50)
print(f"\nBest hyperparameters: alpha={grid_search.best_params_['alpha']}")
print(f"\nValidation Set Performance:")
print(f"  R² Score: {val_r2:.4f}")
print(f"  RMSE: {val_rmse:.2f}")
print(f"  MAE: {val_mae:.2f}")
print(f"\nTest Set Performance:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  RMSE: {test_rmse:.2f}")
print(f"  MAE: {test_mae:.2f}")

# Show feature importance (top 10)
feature_names = dv.get_feature_names_out()
coef_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': best_model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(f"\nTop 10 Most Important Features:")
print(coef_importance.head(10).to_string(index=False))

# Save the model and vectorizer
with open('model.pkl', 'wb') as f_model:
    pickle.dump(best_model, f_model)

with open('vectorizer.pkl', 'wb') as f_vec:
    pickle.dump(dv, f_vec)

print("\n✓ Model saved to 'model.pkl'")
print("✓ Vectorizer saved to 'vectorizer.pkl'")