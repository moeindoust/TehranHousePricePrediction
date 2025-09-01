import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor # Import XGBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import io
import joblib # Import joblib baraye zakhire kardan model va scaler

# func baraye hazf outlier ha ba estefade az IQR
def remove_outliers_iqr(df, column):
    """
    Removes outliers from a specified column in a DataFrame using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f'{column} Lower bound: {lower_bound}, Upper bound: {upper_bound}')
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# load karardan data
# hatamn address file ra qabl az ejra check konid
try:
    df_original = pd.read_csv('C:\\Users\\Doust\\NoteBook\\houseprice_maktabkhone\\housepriceMKclean.csv')
except FileNotFoundError:
    print("Error: 'housepriceMKclean.csv' not found. Please ensure the file is accessible in the correct path.")
    exit()

# joda kardan feature haye mored nazar
cdf = df_original[['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address', 'Price']].copy()

#paksazi
df_no_outliers_area = remove_outliers_iqr(cdf, 'Area')
df_no_outliers = remove_outliers_iqr(df_no_outliers_area, 'Price')

print(f"Tedad record after IQR: {len(df_no_outliers)}")

# --- Feature Engineering and Preprocessing ---
# convert kardan feature haye mantiqi be adadi baraye model XGregr
df_no_outliers['Parking'] = df_no_outliers['Parking'].astype(int)
df_no_outliers['Warehouse'] = df_no_outliers['Warehouse'].astype(int)
df_no_outliers['Elevator'] = df_no_outliers['Elevator'].astype(int)

# Add a new feature: Log_Price
# tabdil be Log Target baraye behtar kardan model 
df_no_outliers['Log_Price'] = np.log1p(df_no_outliers['Price'])

# Encoding...
df_processed = pd.get_dummies(df_no_outliers, columns=['Address'], prefix='Address', dtype=int)

#joda kardan X va y
X = df_processed.drop(columns=['Price', 'Log_Price'])
y = df_processed['Log_Price'] # Set Log_Price as the target

print("\nShape of X (features matrix):", X.shape)
print("Shape of y (target vector):", y.shape)

# Train-Test Split...
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Scaling ---
# Note* : ba inke XGBoost be scaling niyaz nadarad, amma baraye hamvar kardan ba pipeline ha va model haye digar in kar ra aanjam midahim.
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model Training...
# n_estimators: tedad derakht ha.
# random_state: baraye tkrar paziri natayej.
# n_jobs=-1: Uses all available CPU cores for faster training. shayadam lazem nabashe.
model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(x_train_scaled, y_train)

# Prediction and Inverse Transformation...
y_pred_log = model.predict(x_test_scaled)

# baraye barghasht be scale asli
# This is necessary chon ke ma log transform estefade kardim baraye target.
y_pred_original_scale = np.expm1(y_pred_log) 
y_test_original_scale = np.expm1(y_test)   

# berim soragh Evaluation Metrics...
r2 = r2_score(y_test_original_scale, y_pred_original_scale)
mae = mean_absolute_error(y_test_original_scale, y_pred_original_scale)
mse = mean_squared_error(y_test_original_scale, y_pred_original_scale)

print(f"\nUpdated R2 Score (with XGBoostRegressor): {r2}")
print(f"Updated MAE (with XGBoostRegressor): {mae}")
print(f"Updated MSE (with XGBoostRegressor): {mse}")

if hasattr(model, 'feature_importances_'):
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importances (Top 20):")
    print(feature_importances.head(20)) 

# Plot Actual vs Predicted Price ha.
# hacheqadr be nimsaz nazdik bashan data ha behtere.
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_original_scale, y=y_pred_original_scale, alpha=0.6)
plt.plot([y_test_original_scale.min(), y_test_original_scale.max()], [y_test_original_scale.min(), y_test_original_scale.max()], 'r--', lw=2)
plt.xlabel("Actual Price (Toman)")
plt.ylabel("Predicted Price (Toman)")
plt.title("Actual vs Predicted Prices (XGBRegr)")
plt.grid(True)
plt.show()

# upload kardan model va scaler va feature columns baraye estefade dar app.
joblib.dump(model, 'house_price_model.joblib')
print("\nModel saved successfully as 'house_price_model.joblib'")

joblib.dump(scaler, 'scaler.joblib')
print("Scaler saved successfully as 'scaler.joblib'")

joblib.dump(X.columns.tolist(), 'features_columns.joblib')
print("Feature columns saved successfully as 'features_columns.joblib'")

