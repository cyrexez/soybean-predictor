# train.py - Compare multiple regression models
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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

# Combine train and val for cross-validation
X_train_val = np.vstack([X_train, X_val])
y_train_val = np.concatenate([y_train, y_val])

# Define models and their hyperparameter grids
models = {
    'Ridge': {
        'model': Ridge(random_state=42),
        'params': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    },
    'Lasso': {
        'model': Lasso(random_state=42, max_iter=10000),
        'params': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    },
    'ElasticNet': {
        'model': ElasticNet(random_state=42, max_iter=10000),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.2, 0.5, 0.8]
        }
    },
    'Decision Tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }
}

# Train and evaluate all models
results = []
trained_models = {}

print("\n" + "=" * 70)
print("TRAINING AND EVALUATING MULTIPLE REGRESSION MODELS")
print("=" * 70)

for name, config in models.items():
    print(f"\n{'='*70}")
    print(f"Training {name}...")
    print(f"{'='*70}")
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        config['model'],
        config['params'],
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_val, y_train_val)
    best_model = grid_search.best_estimator_
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Store results
    results.append({
        'Model': name,
        'Best_Params': str(grid_search.best_params_),
        'CV_R2': grid_search.best_score_,
        'Val_R2': val_r2,
        'Val_RMSE': val_rmse,
        'Val_MAE': val_mae,
        'Test_R2': test_r2,
        'Test_RMSE': test_rmse,
        'Test_MAE': test_mae
    })
    
    trained_models[name] = best_model
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"CV R² Score: {grid_search.best_score_:.4f}")
    print(f"\nValidation Performance:")
    print(f"  R² Score: {val_r2:.4f}")
    print(f"  RMSE: {val_rmse:.2f}")
    print(f"  MAE: {val_mae:.2f}")
    print(f"\nTest Performance:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE: {test_mae:.2f}")

# Create results dataframe and sort by test R²
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test_R2', ascending=False)

print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY (Sorted by Test R² Score)")
print("=" * 70)
print(results_df[['Model', 'CV_R2', 'Val_R2', 'Test_R2', 'Test_RMSE', 'Test_MAE']].to_string(index=False))

# Select best model based on test R²
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
best_test_r2 = results_df.iloc[0]['Test_R2']

print("\n" + "=" * 70)
print(f"BEST MODEL: {best_model_name}")
print("=" * 70)
print(f"Test R² Score: {best_test_r2:.4f}")
print(f"Test RMSE: {results_df.iloc[0]['Test_RMSE']:.2f}")
print(f"Test MAE: {results_df.iloc[0]['Test_MAE']:.2f}")

# Show feature importance for best model if available
if hasattr(best_model, 'coef_'):
    feature_names = dv.get_feature_names_out()
    coef_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': best_model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\nTop 10 Most Important Features (by coefficient magnitude):")
    print(coef_importance.head(10).to_string(index=False))
    
elif hasattr(best_model, 'feature_importances_'):
    feature_names = dv.get_feature_names_out()
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features (by importance):")
    print(feature_importance.head(10).to_string(index=False))

# Save the best model and vectorizer
with open('model.pkl', 'wb') as f_model:
    pickle.dump(best_model, f_model)

with open('vectorizer.pkl', 'wb') as f_vec:
    pickle.dump(dv, f_vec)

# Save model metadata
metadata = {
    'model_name': best_model_name,
    'test_r2': float(best_test_r2),
    'test_rmse': float(results_df.iloc[0]['Test_RMSE']),
    'test_mae': float(results_df.iloc[0]['Test_MAE']),
    'best_params': results_df.iloc[0]['Best_Params']
}

with open('model_metadata.pkl', 'wb') as f_meta:
    pickle.dump(metadata, f_meta)

print("\n✓ Best model saved to 'model.pkl'")
print("✓ Vectorizer saved to 'vectorizer.pkl'")
print("✓ Model metadata saved to 'model_metadata.pkl'")