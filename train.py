from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from data import get_dataset

CROP_TYPES = ['Barley', 'Corn', 'Potato', 'Rice', 'Soybean', 'Sugarcane', 'Sunflower', 'Wheat']

df = get_dataset()

regressors = {}

for crop in CROP_TYPES:
    crop_data = df[df['Crop_Type'] == crop]
    X = crop_data[['Humidity', 'Temperature']]
    y = crop_data['Crop_Yield']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    param_grid = {
        'n_estimators': [500, 700, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [6, 9, 12],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 5, 10]
    }

    tscv = TimeSeriesSplit(n_splits=5)

    model = XGBRegressor(
        objective='reg:squarederror', 
        early_stopping_rounds=20,
        tree_method='hist',
        enable_categorical=True
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    
    best_model = grid_search.best_estimator_
    regressors[crop] = best_model
    
    y_pred = best_model.predict(X_test)
    print(f'Crop: {crop}')
    print(f'MSE: {mean_squared_error(y_test, y_pred):.2f}')
    print(f'R-Squared: {r2_score(y_test, y_pred):.2f}\n')

joblib.dump(regressors, 'model/regressors.pkl')
