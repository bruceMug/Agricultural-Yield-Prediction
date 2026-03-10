"""
================================================================================
AGRICULTURAL YIELD PREDICTION SYSTEM FOR FARMERS IN KENYA
Part 1: Climate Data Analysis and Rainfall Forecasting
================================================================================

Team 1: Ethel Kwagalakwe | Martin Mulang' | Bruce Mugizi | Emrys Nicholaus
Course: Artificial Intelligence System Design
Institution: Carnegie Mellon University
Date: February 2026

This script performs comprehensive EDA, feature engineering, and model training
for rainfall forecasting in Kenya.

REFERENCES:
[1] Shetty, D. (2024). Food insecurity affects 282 million people in 2023.
[2] Masipa, T. S. (2017). The impact of climate change on food security in South Africa.
[3] Digital Earth Africa (2024). Creating an open-source framework for crop-type mapping.
[4] Hochreiter & Schmidhuber (1997). Long short-term memory.
[5] Breiman (2001). Random forests.
[6] Ke et al. (2017). LightGBM: A highly efficient gradient boosting decision tree.
[7] Lundberg & Lee (2017). A unified approach to interpreting model predictions.
================================================================================
"""

# ==============================================================================
# 1. IMPORT LIBRARIES AND SETUP
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta

# Machine Learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Explainability
import shap

# Statistical Analysis
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("="*80)
print("KENYA RAINFALL PREDICTION SYSTEM - INITIALIZATION")
print("="*80)
print(f"✅ TensorFlow version: {tf.__version__}")
print(f"✅ Pandas version: {pd.__version__}")
print(f"✅ Random seed set: {RANDOM_SEED}")
print()

# ==============================================================================
# 2. DATA LOADING AND INITIAL EXPLORATION
# ==============================================================================

print("="*80)
print("SECTION 2: DATA LOADING")
print("="*80)

# Load the combined dataset
# UPDATE THIS PATH to match your file location
DATA_PATH = 'kenya_climate_combined.csv'

try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data loaded successfully from: {DATA_PATH}")
except FileNotFoundError:
    print(f"❌ Error: File not found at {DATA_PATH}")
    print("Please update DATA_PATH variable with the correct file location.")
    exit(1)

# Convert time column to datetime
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values(['location', 'time']).reset_index(drop=True)

print(f"\n📊 DATASET OVERVIEW:")
print(f"   Total Records: {len(df):,}")
print(f"   Number of Locations: {df['location'].nunique()}")
print(f"   Date Range: {df['time'].min()} to {df['time'].max()}")
print(f"   Time Span: {(df['time'].max() - df['time'].min()).days} days")
print(f"\n   Columns: {list(df.columns)}")
print(f"\n   Data types:")
print(df.dtypes)

# Statistical summary
print(f"\n📈 STATISTICAL SUMMARY:")
print(df.describe())

# Missing values check
print(f"\n🔍 MISSING VALUES:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✅ No missing values found!")
else:
    print(missing[missing > 0])

# Records per location
print(f"\n📍 RECORDS PER LOCATION (Top 10):")
location_counts = df['location'].value_counts().head(10)
for loc, count in location_counts.items():
    print(f"   {loc}: {count:,}")

print()

# ==============================================================================
# 3. AGRO-ECOLOGICAL ZONE CLASSIFICATION
# ==============================================================================

print("="*80)
print("SECTION 3: AGRO-ECOLOGICAL ZONE CLASSIFICATION")
print("="*80)

# Classify locations into agro-ecological zones
# Based on Kenya's climate zones and agricultural potential
zone_mapping = {
    'Coastal': ['Lamu', 'Malindi', 'Mombasa', 'Voi'],
    'Highlands': ['Kitale', 'Eldoret', 'Eldoret_Airstrip', 'Kakamega', 'Kisumu', 
                  'Kericho', 'Kisii', 'Nanyuki', 'Meru', 'Nyeri', 'Embu'],
    'Nairobi': ['Nairobi_ACC', 'Nairobi_Dagoretti', 'Nairobi_Wilson', 'Nairobi_Doonholm'],
    'ASAL': ['Lodwar', 'Moyale', 'Mandera', 'Marsabit', 'Wajir', 'Garissa'],
    'Other': ['Nakuru', 'Narok', 'Makindu', 'Masai_Mara']
}

# Create reverse mapping
location_to_zone = {}
for zone, locations in zone_mapping.items():
    for loc in locations:
        location_to_zone[loc] = zone

df['zone'] = df['location'].map(location_to_zone)

print("✅ Agro-ecological zones assigned:")
for zone, locs in zone_mapping.items():
    print(f"   {zone}: {len(locs)} locations")

# Zone summary statistics
print(f"\n📊 CLIMATE SUMMARY BY ZONE:")
zone_summary = df.groupby('zone').agg({
    'tavg': 'mean',
    'prcp': ['mean', 'sum'],
    'location': 'nunique'
}).round(2)
zone_summary.columns = ['Avg Temp (°C)', 'Mean Daily Rainfall (mm)', 
                        'Total Rainfall (mm)', 'Num Stations']
print(zone_summary)
print()

# ==============================================================================
# 4. TEMPORAL FEATURE EXTRACTION
# ==============================================================================

print("="*80)
print("SECTION 4: TEMPORAL FEATURE EXTRACTION")
print("="*80)

# Extract temporal features
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day_of_year'] = df['time'].dt.dayofyear
df['quarter'] = df['time'].dt.quarter
df['week_of_year'] = df['time'].dt.isocalendar().week

# Define Kenya's agricultural seasons
def get_season(month):
    """
    Classify months into Kenya's agricultural seasons:
    - Long Rains (MAM): March-May - Critical planting season
    - Short Rains (OND): October-December - Secondary planting season
    - Dry Season: June-September
    - Inter-Season: January-February
    """
    if month in [3, 4, 5]:
        return 'Long_Rains'
    elif month in [10, 11, 12]:
        return 'Short_Rains'
    elif month in [6, 7, 8, 9]:
        return 'Dry_Season'
    else:
        return 'Inter_Season'

df['season'] = df['month'].apply(get_season)

print("✅ Temporal features extracted:")
print("   - year, month, day_of_year, quarter, week_of_year")
print("   - season (Agricultural seasons for Kenya)")

print(f"\n📅 SEASONAL DISTRIBUTION:")
season_counts = df['season'].value_counts()
for season, count in season_counts.items():
    print(f"   {season}: {count:,} records")

# Seasonal rainfall statistics
print(f"\n🌧️  SEASONAL RAINFALL STATISTICS:")
seasonal_stats = df.groupby('season')['prcp'].agg(['mean', 'std', 'min', 'max']).round(2)
seasonal_stats.columns = ['Mean (mm)', 'Std Dev (mm)', 'Min (mm)', 'Max (mm)']
print(seasonal_stats)
print()

# ==============================================================================
# 5. FEATURE ENGINEERING
# ==============================================================================

print("="*80)
print("SECTION 5: FEATURE ENGINEERING")
print("="*80)
print("Creating features for rainfall forecasting...")
print("This includes lag features, rolling statistics, agricultural indices, etc.")
print()

def engineer_features(data, location=None):
    """
    Create comprehensive features for rainfall forecasting.
    
    Features Created:
    -----------------
    1. LAG FEATURES: Previous 1, 3, 7, 14, 30 days
       - Captures short to medium-term temporal dependencies
       - Essential for LSTM and autoregressive models
    
    2. ROLLING WINDOW STATISTICS: 7, 14, 30, 90 day windows
       - Mean, sum, and standard deviation of rainfall
       - Smooths daily noise
       - Represents soil moisture memory effects
    
    3. AGRICULTURAL INDICES:
       - Growing Degree Days (GDD): Heat accumulation for crop growth
       - Dry Spell Duration: Consecutive days with <1mm rainfall
       - Moisture Availability Index: Proxy for plant-available water
    
    4. CYCLICAL FEATURES:
       - Sine/cosine encoding of month and day_of_year
       - Preserves cyclical nature of seasons
    
    5. EXTREME EVENT INDICATORS:
       - Heavy rain (>50mm/day), moderate, light, no rain
       - Heat stress (>35°C), cold stress (<10°C)
    
    6. INTERACTION FEATURES:
       - Temperature × Rainfall
       - Wind × Pressure
    
    7. RATE OF CHANGE:
       - Daily changes in rainfall, temperature, pressure
    """
    df_features = data.copy()
    
    if location:
        df_features = df_features[df_features['location'] == location].copy()
    
    print(f"Engineering features for: {location if location else 'all locations'}")
    
    # 1. LAG FEATURES
    print("  [1/7] Creating lag features...")
    lag_days = [1, 3, 7, 14, 30]
    for lag in lag_days:
        df_features[f'prcp_lag_{lag}'] = df_features.groupby('location')['prcp'].shift(lag)
        df_features[f'tavg_lag_{lag}'] = df_features.groupby('location')['tavg'].shift(lag)
    
    # 2. ROLLING WINDOW STATISTICS
    print("  [2/7] Creating rolling window statistics...")
    windows = [7, 14, 30, 90]
    for window in windows:
        df_features[f'prcp_rolling_mean_{window}d'] = df_features.groupby('location')['prcp'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df_features[f'prcp_rolling_sum_{window}d'] = df_features.groupby('location')['prcp'].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )
        df_features[f'prcp_rolling_std_{window}d'] = df_features.groupby('location')['prcp'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        df_features[f'tavg_rolling_mean_{window}d'] = df_features.groupby('location')['tavg'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    # 3. AGRICULTURAL INDICES
    print("  [3/7] Creating agricultural indices...")
    # Growing Degree Days (base temperature 10°C)
    df_features['gdd'] = df_features.apply(
        lambda row: max(0, ((row['tmax'] + row['tmin']) / 2) - 10), axis=1
    )
    df_features['gdd_cumsum_30d'] = df_features.groupby('location')['gdd'].transform(
        lambda x: x.rolling(30, min_periods=1).sum()
    )
    
    # Dry spell duration
    def calculate_dry_spell(group):
        dry_spell = (group['prcp'] < 1).astype(int)
        dry_spell_count = dry_spell.groupby((dry_spell != dry_spell.shift()).cumsum()).cumsum()
        return dry_spell_count * dry_spell
    
    df_features['dry_spell_days'] = df_features.groupby('location', group_keys=False).apply(
        calculate_dry_spell
    ).values
    
    # Moisture Availability Index
    df_features['moisture_index'] = df_features['prcp'] / (df_features['prcp_rolling_mean_30d'] + 0.1)
    
    # Temperature range (diurnal variation)
    df_features['temp_range'] = df_features['tmax'] - df_features['tmin']
    
    # 4. CYCLICAL FEATURES
    print("  [4/7] Creating cyclical features...")
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
    df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
    
    # 5. EXTREME EVENT INDICATORS
    print("  [5/7] Creating extreme event indicators...")
    df_features['heavy_rain'] = (df_features['prcp'] > 50).astype(int)
    df_features['moderate_rain'] = ((df_features['prcp'] > 10) & (df_features['prcp'] <= 50)).astype(int)
    df_features['light_rain'] = ((df_features['prcp'] > 1) & (df_features['prcp'] <= 10)).astype(int)
    df_features['no_rain'] = (df_features['prcp'] <= 1).astype(int)
    df_features['heat_stress'] = (df_features['tmax'] > 35).astype(int)
    df_features['cold_stress'] = (df_features['tmin'] < 10).astype(int)
    
    # 6. INTERACTION FEATURES
    print("  [6/7] Creating interaction features...")
    df_features['temp_prcp_interaction'] = df_features['tavg'] * df_features['prcp']
    df_features['wind_pressure_interaction'] = df_features['wspd'] * df_features['pres']
    
    # 7. RATE OF CHANGE FEATURES
    print("  [7/7] Creating rate of change features...")
    df_features['prcp_change'] = df_features.groupby('location')['prcp'].diff()
    df_features['temp_change'] = df_features.groupby('location')['tavg'].diff()
    df_features['pressure_change'] = df_features.groupby('location')['pres'].diff()
    
    print(f"  ✅ Feature engineering complete! Total features: {len(df_features.columns)}")
    return df_features

# Apply feature engineering
df_engineered = engineer_features(df)

print(f"\n📊 FEATURE ENGINEERING SUMMARY:")
print(f"   Original features: {len(df.columns)}")
print(f"   Total features after engineering: {len(df_engineered.columns)}")
print(f"   New features created: {len(df_engineered.columns) - len(df.columns)}")
print()

# ==============================================================================
# 6. DATA PREPARATION FOR MODELING
# ==============================================================================

print("="*80)
print("SECTION 6: DATA PREPARATION FOR MODELING")
print("="*80)

# Remove rows with missing values
df_clean = df_engineered.dropna().copy()
print(f"Dataset size after removing NaN: {len(df_clean):,} records")
print(f"Removed {len(df_engineered) - len(df_clean):,} records with missing values")

# Select a representative location for modeling
# Nairobi_ACC has the most complete data
target_location = 'Nairobi_ACC'
df_model = df_clean[df_clean['location'] == target_location].copy()
df_model = df_model.sort_values('time').reset_index(drop=True)

print(f"\n📍 TARGET LOCATION: {target_location}")
print(f"   Total records: {len(df_model):,}")
print(f"   Date range: {df_model['time'].min()} to {df_model['time'].max()}")
print(f"   Duration: {(df_model['time'].max() - df_model['time'].min()).days} days")

# Define features for modeling
exclude_modeling = ['time', 'location', 'station_id', 'zone', 'season', 
                    'year', 'month', 'day_of_year', 'quarter', 'week_of_year']

feature_columns = [col for col in df_model.columns if col not in exclude_modeling]
print(f"\n   Total features for modeling: {len(feature_columns)}")

# Create target variables: future rainfall
df_model['target_prcp_1d'] = df_model['prcp'].shift(-1)   # Next day
df_model['target_prcp_3d'] = df_model['prcp'].shift(-3)   # 3 days ahead
df_model['target_prcp_7d'] = df_model['prcp'].shift(-7)   # 7 days ahead

# Remove rows with NaN targets
df_model = df_model.dropna(subset=['target_prcp_1d', 'target_prcp_3d', 'target_prcp_7d'])

print(f"\n   Final modeling dataset: {len(df_model):,} records")
print(f"   Target variables:")
print(f"     - target_prcp_1d: Next day rainfall")
print(f"     - target_prcp_3d: 3-day ahead rainfall")
print(f"     - target_prcp_7d: 7-day ahead rainfall")

# Train-Test Split (temporal split - no shuffling!)
train_size = int(len(df_model) * 0.8)
train_df = df_model.iloc[:train_size].copy()
test_df = df_model.iloc[train_size:].copy()

print(f"\n✂️  TRAIN-TEST SPLIT (Temporal):")
print(f"   Training: {len(train_df):,} records ({train_df['time'].min()} to {train_df['time'].max()})")
print(f"   Test: {len(test_df):,} records ({test_df['time'].min()} to {test_df['time'].max()})")
print(f"   Ratio: {len(train_df)/len(test_df):.2f}")

# Prepare feature matrices and target vectors
X_train = train_df[feature_columns]
X_test = test_df[feature_columns]
y_train_1d = train_df['target_prcp_1d']
y_test_1d = test_df['target_prcp_1d']

print(f"\n   Feature matrix shape: {X_train.shape}")
print(f"   Target vector shape: {y_train_1d.shape}")

# Feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_1d_scaled = scaler_y.fit_transform(y_train_1d.values.reshape(-1, 1)).ravel()
y_test_1d_scaled = scaler_y.transform(y_test_1d.values.reshape(-1, 1)).ravel()

print(f"\n   ✅ Data preparation complete!")
print(f"   Scaled features range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
print()

# ==============================================================================
# 7. MODEL TRAINING - RANDOM FOREST
# ==============================================================================

print("="*80)
print("SECTION 7: RANDOM FOREST REGRESSION")
print("="*80)

def evaluate_model(y_true, y_pred, model_name, dataset_name):
    """Calculate and display evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} - {dataset_name}:")
    print(f"  RMSE: {rmse:.4f} mm")
    print(f"  MAE:  {mae:.4f} mm")
    print(f"  R²:   {r2:.4f}")
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

print("Training Random Forest baseline model...")
print("Hyperparameters: n_estimators=100, max_depth=None, min_samples_split=2")

rf_baseline = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=0
)

rf_baseline.fit(X_train, y_train_1d)
print("✅ Training complete!")

# Predictions
y_pred_train_rf = rf_baseline.predict(X_train)
y_pred_test_rf = rf_baseline.predict(X_test)

# Evaluation
rf_train_metrics = evaluate_model(y_train_1d, y_pred_train_rf, "Random Forest Baseline", "Training Set")
rf_test_metrics = evaluate_model(y_test_1d, y_pred_test_rf, "Random Forest Baseline", "Test Set")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_baseline.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📊 TOP 15 MOST IMPORTANT FEATURES:")
print(feature_importance.head(15).to_string(index=False))
print()

# ==============================================================================
# 8. MODEL TRAINING - LIGHTGBM
# ==============================================================================

print("="*80)
print("SECTION 8: LIGHTGBM REGRESSION")
print("="*80)

print("Training LightGBM baseline model...")
print("Hyperparameters: num_leaves=31, learning_rate=0.05, n_estimators=100")

lgb_baseline = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
    random_state=RANDOM_SEED,
    verbose=-1
)

lgb_baseline.fit(
    X_train, y_train_1d,
    eval_set=[(X_test, y_test_1d)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
)

print(f"✅ Training complete! (Best iteration: {lgb_baseline.best_iteration_})")

# Predictions
y_pred_test_lgb = lgb_baseline.predict(X_test)

# Evaluation
lgb_test_metrics = evaluate_model(y_test_1d, y_pred_test_lgb, "LightGBM Baseline", "Test Set")
print()

# ==============================================================================
# 9. MODEL TRAINING - LSTM
# ==============================================================================

print("="*80)
print("SECTION 9: LSTM NEURAL NETWORK")
print("="*80)

def create_sequences(X, y, sequence_length=30):
    """Create sequences for LSTM training"""
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    return np.array(X_seq), np.array(y_seq)

SEQUENCE_LENGTH = 30
print(f"Creating sequences with lookback window: {SEQUENCE_LENGTH} days")

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_1d_scaled, SEQUENCE_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_1d_scaled, SEQUENCE_LENGTH)

print(f"  Training sequences: {X_train_seq.shape} (samples, timesteps, features)")
print(f"  Test sequences: {X_test_seq.shape}")

# Build LSTM model
print("\nBuilding LSTM architecture...")
print("  Layer 1: Bidirectional LSTM (128 units)")
print("  Layer 2: Bidirectional LSTM (64 units)")
print("  Layer 3: Dense (32 units, ReLU)")
print("  Layer 4: Dense (1 unit, Linear)")

lstm_model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, activation='tanh'),
                  input_shape=(SEQUENCE_LENGTH, X_train_seq.shape[2])),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False, activation='tanh')),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.15),
    Dense(1, activation='linear')
], name='LSTM_Rainfall_Forecaster')

lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

trainable_params = np.sum([np.prod(v.get_shape()) for v in lstm_model.trainable_weights])
print(f"\n  Total trainable parameters: {trainable_params:,}")

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=0
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=0
)

# Training
print("\nTraining LSTM model...")
print("  Batch size: 64")
print("  Max epochs: 100 (with early stopping)")
print("  Validation split: 20%")

history = lstm_model.fit(
    X_train_seq, y_train_seq,
    batch_size=64,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

print(f"✅ Training complete! (Best epoch: {early_stop.best_epoch + 1})")

# Predictions
y_pred_test_lstm_scaled = lstm_model.predict(X_test_seq, verbose=0)
y_pred_test_lstm = scaler_y.inverse_transform(y_pred_test_lstm_scaled).ravel()
y_test_actual = y_test_1d.values[SEQUENCE_LENGTH:]

# Evaluation
lstm_test_metrics = evaluate_model(y_test_actual, y_pred_test_lstm, "LSTM", "Test Set")
print()

# ==============================================================================
# 10. MODEL COMPARISON
# ==============================================================================

print("="*80)
print("SECTION 10: MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'LightGBM', 'LSTM'],
    'RMSE (mm)': [
        rf_test_metrics['RMSE'],
        lgb_test_metrics['RMSE'],
        lstm_test_metrics['RMSE']
    ],
    'MAE (mm)': [
        rf_test_metrics['MAE'],
        lgb_test_metrics['MAE'],
        lstm_test_metrics['MAE']
    ],
    'R²': [
        rf_test_metrics['R2'],
        lgb_test_metrics['R2'],
        lstm_test_metrics['R2']
    ]
})

print("\n📊 MODEL PERFORMANCE COMPARISON (Test Set):")
print(comparison_df.to_string(index=False))

# Find best model
best_rmse_idx = comparison_df['RMSE (mm)'].idxmin()
best_r2_idx = comparison_df['R²'].idxmax()

print(f"\n🏆 BEST MODELS:")
print(f"   Lowest RMSE: {comparison_df.loc[best_rmse_idx, 'Model']} ({comparison_df.loc[best_rmse_idx, 'RMSE (mm)']:.4f} mm)")
print(f"   Highest R²:  {comparison_df.loc[best_r2_idx, 'Model']} ({comparison_df.loc[best_r2_idx, 'R²']:.4f})")
print()

# ==============================================================================
# 11. SAVE RESULTS
# ==============================================================================

print("="*80)
print("SECTION 11: SAVING RESULTS")
print("="*80)

# Save models
import joblib

try:
    joblib.dump(rf_baseline, 'random_forest_model.pkl')
    joblib.dump(lgb_baseline, 'lightgbm_model.pkl')
    lstm_model.save('lstm_model.h5')
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    print("✅ Models saved successfully:")
    print("   - random_forest_model.pkl")
    print("   - lightgbm_model.pkl")
    print("   - lstm_model.h5")
    print("   - scaler_X.pkl")
    print("   - scaler_y.pkl")
except Exception as e:
    print(f"❌ Error saving models: {e}")

# Save comparison results
comparison_df.to_csv('model_comparison_results.csv', index=False)
print("   - model_comparison_results.csv")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("   - feature_importance.csv")

print()
print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\n✅ All tasks completed successfully!")
print("\nNext steps:")
print("  1. Review model performance metrics")
print("  2. Analyze feature importance")
print("  3. Test models on other locations")
print("  4. Integrate with agricultural yield data")
print("  5. Deploy for SMS-based farmer advisory system")
print()
