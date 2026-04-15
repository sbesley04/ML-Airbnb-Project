#!/usr/bin/env python3
"""
NYC Airbnb Pricing Prediction Analysis Pipeline

Comprehensive machine learning analysis for predicting NYC Airbnb prices using
natural language processing on listing names and multiple regression models.

QTM 347 (Machine Learning) - Emory University
Team: Connor Lee, Kaya Monrose, Parker Shimp, Sam Besley

Author: Sam Besley
Date: 2026-04-15
"""

# ==============================================================================
# REQUIREMENTS
# ==============================================================================
"""
Required packages (install via: pip install -r requirements.txt):
    pandas>=1.3.0
    numpy>=1.21.0
    scikit-learn>=0.24.0
    matplotlib>=3.4.0
    seaborn>=0.11.0
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Project paths (relative to script location)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Ensure output directories exist
FIGURES_DIR.mkdir(exist_ok=True)

# Analysis parameters
RANDOM_STATE = 42
TRAIN_TEST_SPLIT = 0.8
PRICE_MIN = 0
PRICE_MAX = 1000
LOG_TRANSFORM_TARGET = True

# Luxury and budget keywords for NLP feature engineering
LUXURY_KEYWORDS = [
    'luxury', 'spacious', 'modern', 'penthouse', 'elegant',
    'sophisticated', 'upscale', 'premium', 'stunning', 'beautiful',
    'gorgeous', 'lavish', 'exclusive', 'designer', 'state-of-the-art'
]

BUDGET_KEYWORDS = [
    'cozy', 'small', 'tiny', 'budget', 'affordable', 'simple',
    'basic', 'modest', 'intimate', 'quaint', 'compact'
]

LOCATION_KEYWORDS = [
    'central', 'midtown', 'subway', 'downtown', 'uptown', 'manhattan',
    'brooklyn', 'queens', 'bronx', 'harlem', 'soho', 'tribeca', 'williamsburg'
]


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Airbnb dataset from CSV file.

    Args:
        filepath: Path to the AB_NYC_2019.csv file

    Returns:
        DataFrame with raw Airbnb data
    """
    df = pd.read_csv(filepath)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def display_basic_stats(df: pd.DataFrame) -> None:
    """
    Display basic statistics about the dataset.

    Args:
        df: Input DataFrame
    """
    print("\n" + "="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    print(f"\nShape: {df.shape}")
    print(f"\nColumn Information:")
    print(df.info())
    print(f"\nBasic Statistics:")
    print(df.describe())
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nPrice Range: ${df['price'].min()} - ${df['price'].max()}")
    print(f"Room Types: {df['room_type'].unique()}")
    print(f"Neighbourhoods Groups: {df['neighbourhood_group'].unique()}")


# ==============================================================================
# DATA CLEANING
# ==============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset.

    Removes invalid prices, fills missing values, and ensures data quality.

    Args:
        df: Raw input DataFrame

    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()

    print("\n" + "="*80)
    print("DATA CLEANING")
    print("="*80)

    # Remove invalid prices
    initial_count = len(df_clean)
    df_clean = df_clean[(df_clean['price'] > PRICE_MIN) & (df_clean['price'] <= PRICE_MAX)]
    removed_count = initial_count - len(df_clean)
    print(f"\nRemoved {removed_count} rows with invalid prices (outside ${PRICE_MIN}-${PRICE_MAX})")
    print(f"Remaining rows: {len(df_clean)}")

    # Fill missing reviews_per_month with 0
    missing_reviews = df_clean['reviews_per_month'].isnull().sum()
    df_clean['reviews_per_month'].fillna(0, inplace=True)
    print(f"Filled {missing_reviews} missing 'reviews_per_month' values with 0")

    # Fill missing names with generic placeholder
    missing_names = df_clean['name'].isnull().sum()
    df_clean['name'].fillna("Unnamed Listing", inplace=True)
    print(f"Filled {missing_names} missing 'name' values with 'Unnamed Listing'")

    # Convert price to log scale
    if LOG_TRANSFORM_TARGET:
        df_clean['price_log'] = np.log1p(df_clean['price'])
        print(f"\nLog-transformed price as target variable")

    return df_clean


# ==============================================================================
# NLP FEATURE ENGINEERING
# ==============================================================================

def engineer_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer NLP features from listing names.

    Creates linguistic and semantic features:
    - name_length: Character count in listing name
    - name_word_count: Number of words in name
    - name_has_exclamation: Binary indicator for exclamation marks
    - name_all_caps_ratio: Proportion of words in all caps
    - name_luxury_count: Count of luxury-related keywords
    - name_budget_count: Count of budget-related keywords
    - name_location_count: Count of location-related keywords
    - name_has_bedroom_num: Binary indicator for bedroom numbers

    Args:
        df: Input DataFrame with 'name' column

    Returns:
        DataFrame with added NLP features
    """
    df_nlp = df.copy()

    print("\n" + "="*80)
    print("NLP FEATURE ENGINEERING")
    print("="*80)

    # Basic text features
    df_nlp['name_length'] = df_nlp['name'].str.len()
    df_nlp['name_word_count'] = df_nlp['name'].str.split().str.len()
    df_nlp['name_has_exclamation'] = df_nlp['name'].str.contains('!', regex=False).astype(int)

    # All caps ratio
    def calc_caps_ratio(name):
        if len(name) == 0:
            return 0
        words = str(name).split()
        if len(words) == 0:
            return 0
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        return caps_words / len(words)

    df_nlp['name_all_caps_ratio'] = df_nlp['name'].apply(calc_caps_ratio)

    # Luxury keywords
    def count_keywords(name, keywords):
        name_lower = str(name).lower()
        return sum(1 for kw in keywords if kw.lower() in name_lower)

    df_nlp['name_luxury_count'] = df_nlp['name'].apply(
        lambda x: count_keywords(x, LUXURY_KEYWORDS)
    )
    df_nlp['name_budget_count'] = df_nlp['name'].apply(
        lambda x: count_keywords(x, BUDGET_KEYWORDS)
    )
    df_nlp['name_location_count'] = df_nlp['name'].apply(
        lambda x: count_keywords(x, LOCATION_KEYWORDS)
    )

    # Bedroom number indicator
    df_nlp['name_has_bedroom_num'] = df_nlp['name'].str.contains(
        r'\d+\s*(?:bedroom|bed|br|bd)', regex=True, case=False, na=False
    ).astype(int)

    print(f"Created 8 NLP features from listing names")
    print(f"  - name_length: avg={df_nlp['name_length'].mean():.1f}")
    print(f"  - name_word_count: avg={df_nlp['name_word_count'].mean():.1f}")
    print(f"  - name_has_exclamation: {df_nlp['name_has_exclamation'].sum()} listings with '!'")
    print(f"  - name_all_caps_ratio: avg={df_nlp['name_all_caps_ratio'].mean():.3f}")
    print(f"  - name_luxury_count: avg={df_nlp['name_luxury_count'].mean():.2f}")
    print(f"  - name_budget_count: avg={df_nlp['name_budget_count'].mean():.2f}")
    print(f"  - name_location_count: avg={df_nlp['name_location_count'].mean():.2f}")
    print(f"  - name_has_bedroom_num: {df_nlp['name_has_bedroom_num'].sum()} listings with bedroom numbers")

    return df_nlp


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Engineer features for modeling including encoding and transformations.

    Creates additional engineered features:
    - is_high_availability: Binary for availability > 300 days
    - reviews_per_month_sq: Squared term for reviews
    - log_min_nights: Log-transformed minimum nights
    - log_host_listings: Log-transformed host listings count
    - One-hot encoded room_type and neighbourhood_group

    Args:
        df: Input DataFrame with NLP features

    Returns:
        Tuple of (feature-engineered DataFrame, numeric features list, categorical features list)
    """
    df_feat = df.copy()

    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)

    # Engineered features
    df_feat['is_high_availability'] = (df_feat['availability_365'] > 300).astype(int)
    df_feat['reviews_per_month_sq'] = df_feat['reviews_per_month'] ** 2
    df_feat['log_min_nights'] = np.log1p(df_feat['minimum_nights'])
    df_feat['log_host_listings'] = np.log1p(df_feat['calculated_host_listings_count'])

    print(f"Created 4 engineered features:")
    print(f"  - is_high_availability: {df_feat['is_high_availability'].sum()} high-availability listings")
    print(f"  - reviews_per_month_sq: squared term for regularization")
    print(f"  - log_min_nights: log-transformed minimum nights")
    print(f"  - log_host_listings: log-transformed host listings count")

    # Select features for modeling
    nlp_features = [
        'name_length', 'name_word_count', 'name_has_exclamation',
        'name_all_caps_ratio', 'name_luxury_count', 'name_budget_count',
        'name_location_count', 'name_has_bedroom_num'
    ]

    numeric_features = (
        [
            'latitude', 'longitude', 'minimum_nights',
            'number_of_reviews', 'reviews_per_month',
            'calculated_host_listings_count', 'availability_365'
        ] +
        nlp_features +
        [
            'is_high_availability', 'reviews_per_month_sq',
            'log_min_nights', 'log_host_listings'
        ]
    )

    categorical_features = ['room_type', 'neighbourhood_group']

    print(f"\nSelected features:")
    print(f"  - {len(numeric_features)} numeric features")
    print(f"  - {len(categorical_features)} categorical features")

    return df_feat, numeric_features, categorical_features


# ==============================================================================
# PREPROCESSING & TRAIN/TEST SPLIT
# ==============================================================================

def prepare_modeling_data(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for modeling: encode, split, and standardize.

    Args:
        df: Feature-engineered DataFrame
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    print("\n" + "="*80)
    print("DATA PREPARATION FOR MODELING")
    print("="*80)

    # One-hot encode categorical features
    df_encoded = df.copy()
    categorical_encoded = pd.get_dummies(
        df_encoded[categorical_features],
        drop_first=True,
        prefix=categorical_features
    )

    # Combine all features
    X = pd.concat(
        [df_encoded[numeric_features], categorical_encoded],
        axis=1
    )

    # Target variable
    y = df_encoded['price_log'].values if LOG_TRANSFORM_TARGET else df_encoded['price'].values

    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=1 - TRAIN_TEST_SPLIT,
        random_state=RANDOM_STATE
    )

    print(f"\nTrain/Test Split: {TRAIN_TEST_SPLIT:.0%} / {1-TRAIN_TEST_SPLIT:.0%}")
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nFeatures standardized (mean=0, std=1)")

    return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns)


# ==============================================================================
# MODEL TRAINING & EVALUATION
# ==============================================================================

def train_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str]
) -> Dict:
    """
    Train multiple regression models and evaluate performance.

    Models:
    - Ordinary Least Squares (OLS)
    - Ridge Regression (multiple alpha values)
    - LASSO Regression
    - Random Forest Regressor

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        y_train: Training target
        y_test: Test target
        feature_names: List of feature names

    Returns:
        Dictionary containing models and their metrics
    """
    print("\n" + "="*80)
    print("MODEL TRAINING & EVALUATION")
    print("="*80)

    results = {
        'models': {},
        'metrics': {},
        'feature_names': feature_names
    }

    # 1. OLS (Linear Regression)
    print("\n[1] Ordinary Least Squares (OLS)")
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    y_pred_ols = ols.predict(X_test)

    ols_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ols))
    ols_r2 = r2_score(y_test, y_pred_ols)
    ols_mae = mean_absolute_error(y_test, y_pred_ols)

    results['models']['OLS'] = ols
    results['metrics']['OLS'] = {
        'RMSE': ols_rmse,
        'R²': ols_r2,
        'MAE': ols_mae,
        'predictions': y_pred_ols
    }

    print(f"  RMSE: {ols_rmse:.4f}")
    print(f"  R²:   {ols_r2:.4f}")
    print(f"  MAE:  {ols_mae:.4f}")

    # 2. Ridge Regression (multiple alphas)
    print("\n[2] Ridge Regression")
    ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_ridge_rmse = float('inf')
    best_ridge_model = None
    best_ridge_alpha = None

    for alpha in ridge_alphas:
        ridge = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
        r2 = r2_score(y_test, y_pred_ridge)
        mae = mean_absolute_error(y_test, y_pred_ridge)

        print(f"  α={alpha:6.3f}: RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}")

        if rmse < best_ridge_rmse:
            best_ridge_rmse = rmse
            best_ridge_model = ridge
            best_ridge_alpha = alpha
            best_ridge_metrics = {'RMSE': rmse, 'R²': r2, 'MAE': mae, 'predictions': y_pred_ridge}

    results['models']['Ridge'] = best_ridge_model
    results['models']['Ridge_alpha'] = best_ridge_alpha
    results['metrics']['Ridge'] = best_ridge_metrics

    print(f"  Best α: {best_ridge_alpha} with RMSE={best_ridge_rmse:.4f}")

    # 3. LASSO Regression
    print("\n[3] LASSO Regression")
    lasso = Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)

    lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
    lasso_r2 = r2_score(y_test, y_pred_lasso)
    lasso_mae = mean_absolute_error(y_test, y_pred_lasso)

    results['models']['LASSO'] = lasso
    results['metrics']['LASSO'] = {
        'RMSE': lasso_rmse,
        'R²': lasso_r2,
        'MAE': lasso_mae,
        'predictions': y_pred_lasso
    }

    print(f"  RMSE: {lasso_rmse:.4f}")
    print(f"  R²:   {lasso_r2:.4f}")
    print(f"  MAE:  {lasso_mae:.4f}")

    # 4. Random Forest Regressor
    print("\n[4] Random Forest Regressor")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)

    results['models']['RandomForest'] = rf
    results['metrics']['RandomForest'] = {
        'RMSE': rf_rmse,
        'R²': rf_r2,
        'MAE': rf_mae,
        'predictions': y_pred_rf
    }

    print(f"  RMSE: {rf_rmse:.4f}")
    print(f"  R²:   {rf_r2:.4f}")
    print(f"  MAE:  {rf_mae:.4f}")

    # Summary
    print("\n" + "-"*80)
    print("MODEL COMPARISON SUMMARY")
    print("-"*80)
    for model_name in ['OLS', 'Ridge', 'LASSO', 'RandomForest']:
        metrics = results['metrics'][model_name]
        print(f"{model_name:15s}: RMSE={metrics['RMSE']:.4f}, R²={metrics['R²']:.4f}, MAE={metrics['MAE']:.4f}")

    results['y_test'] = y_test

    return results


# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

def plot_price_distributions(df: pd.DataFrame) -> None:
    """
    Plot raw and log-transformed price distributions.

    Args:
        df: Cleaned DataFrame with price and price_log columns
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw price distribution
    axes[0].hist(df['price'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Price ($)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Raw Price Distribution', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Log-transformed price distribution
    axes[1].hist(df['price_log'], bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_xlabel('Log(Price)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Log-Transformed Price Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '01_price_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved: 01_price_distributions.png")
    plt.close()


def plot_price_by_borough(df: pd.DataFrame) -> None:
    """
    Plot price distribution by borough (neighbourhood_group).

    Args:
        df: Input DataFrame with price and neighbourhood_group columns
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create boxplot
    borough_order = df.groupby('neighbourhood_group')['price'].median().sort_values(ascending=False).index
    sns.boxplot(
        data=df,
        x='neighbourhood_group',
        y='price',
        order=borough_order,
        palette='Set2',
        ax=ax
    )

    ax.set_xlabel('Borough', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_title('Price Distribution by Borough', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 500)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '02_price_by_borough.png', dpi=300, bbox_inches='tight')
    print("Saved: 02_price_by_borough.png")
    plt.close()


def plot_price_by_room_type(df: pd.DataFrame) -> None:
    """
    Plot price distribution by room type.

    Args:
        df: Input DataFrame with price and room_type columns
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create boxplot
    room_order = df.groupby('room_type')['price'].median().sort_values(ascending=False).index
    sns.boxplot(
        data=df,
        x='room_type',
        y='price',
        order=room_order,
        palette='husl',
        ax=ax
    )

    ax.set_xlabel('Room Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_title('Price Distribution by Room Type', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 600)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '03_price_by_room_type.png', dpi=300, bbox_inches='tight')
    print("Saved: 03_price_by_room_type.png")
    plt.close()


def plot_geographic_scatter(df: pd.DataFrame) -> None:
    """
    Plot geographic scatter colored by price.

    Args:
        df: Input DataFrame with latitude, longitude, and price columns
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        df['longitude'],
        df['latitude'],
        c=df['price'],
        s=20,
        alpha=0.6,
        cmap='viridis',
        edgecolors='none'
    )

    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title('NYC Airbnb Listings - Geographic Distribution', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Price ($)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '04_geographic_scatter.png', dpi=300, bbox_inches='tight')
    print("Saved: 04_geographic_scatter.png")
    plt.close()


def plot_top_neighborhoods(df: pd.DataFrame) -> None:
    """
    Plot top 15 neighborhoods by median price.

    Args:
        df: Input DataFrame with neighbourhood and price columns
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    top_neighborhoods = (
        df.groupby('neighbourhood')['price']
        .median()
        .nlargest(15)
        .sort_values(ascending=True)
    )

    ax.barh(range(len(top_neighborhoods)), top_neighborhoods.values, color='teal', alpha=0.7)
    ax.set_yticks(range(len(top_neighborhoods)))
    ax.set_yticklabels(top_neighborhoods.index, fontsize=10)
    ax.set_xlabel('Median Price ($)', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Neighborhoods by Median Price', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for i, v in enumerate(top_neighborhoods.values):
        ax.text(v + 5, i, f'${v:.0f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '05_top_neighborhoods.png', dpi=300, bbox_inches='tight')
    print("Saved: 05_top_neighborhoods.png")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, numeric_features: List[str]) -> None:
    """
    Plot correlation heatmap for numeric features and price.

    Args:
        df: Input DataFrame
        numeric_features: List of numeric feature names
    """
    # Select features and target
    cols_to_plot = numeric_features + ['price']
    corr_matrix = df[cols_to_plot].corr()

    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(
        corr_matrix,
        annot=False,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Correlation'}
    )

    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '06_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: 06_correlation_heatmap.png")
    plt.close()


def plot_model_comparison(results: Dict) -> None:
    """
    Plot model performance comparison.

    Args:
        results: Results dictionary from train_models
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = ['OLS', 'Ridge', 'LASSO', 'RandomForest']
    rmse_scores = [results['metrics'][m]['RMSE'] for m in models]
    r2_scores = [results['metrics'][m]['R²'] for m in models]
    mae_scores = [results['metrics'][m]['MAE'] for m in models]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    # RMSE
    axes[0].bar(models, rmse_scores, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('RMSE', fontsize=11, fontweight='bold')
    axes[0].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(rmse_scores):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

    # R²
    axes[1].bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('R² Score', fontsize=11, fontweight='bold')
    axes[1].set_title('R² Score (Higher is Better)', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, max(r2_scores) * 1.15)
    for i, v in enumerate(r2_scores):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

    # MAE
    axes[2].bar(models, mae_scores, color=colors, alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('MAE ($)', fontsize=11, fontweight='bold')
    axes[2].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(mae_scores):
        axes[2].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '07_model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: 07_model_comparison.png")
    plt.close()


def plot_predicted_vs_actual(results: Dict) -> None:
    """
    Plot predicted vs actual prices for best model (Random Forest).

    Args:
        results: Results dictionary from train_models
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    models = ['OLS', 'Ridge', 'LASSO', 'RandomForest']
    y_test = results['y_test']

    for idx, model_name in enumerate(models):
        y_pred = results['metrics'][model_name]['predictions']

        axes[idx].scatter(y_test, y_pred, alpha=0.5, s=20, color='steelblue', edgecolors='none')

        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        axes[idx].set_xlabel('Actual Log(Price)', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Predicted Log(Price)', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

    plt.suptitle('Predicted vs Actual Prices', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '08_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    print("Saved: 08_predicted_vs_actual.png")
    plt.close()


def plot_feature_importance(results: Dict) -> None:
    """
    Plot feature importance from Ridge regression coefficients.

    Args:
        results: Results dictionary from train_models
    """
    ridge_model = results['models']['Ridge']
    feature_names = results['feature_names']

    # Get feature importance (absolute coefficients)
    importances = np.abs(ridge_model.coef_)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(range(len(importance_df)), importance_df['importance'].values, color='darkgreen', alpha=0.7)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'].values, fontsize=10)
    ax.set_xlabel('|Coefficient| (Importance)', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Most Important Features (Ridge Regression)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for i, v in enumerate(importance_df['importance'].values):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '09_feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: 09_feature_importance.png")
    plt.close()


def plot_nlp_features_analysis(df: pd.DataFrame) -> None:
    """
    Plot NLP features analysis.

    Args:
        df: Input DataFrame with NLP features and price
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Name length vs price
    axes[0, 0].scatter(df['name_length'], df['price'], alpha=0.4, s=15, color='purple')
    axes[0, 0].set_xlabel('Name Length (characters)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Price ($)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Listing Name Length vs Price', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim(0, 500)
    axes[0, 0].grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(df['name_length'], df['price'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['name_length'].sort_values(), p(df['name_length'].sort_values()),
                     "r--", alpha=0.8, linewidth=2, label='Trend')
    axes[0, 0].legend()

    # 2. Luxury words vs price
    axes[0, 1].scatter(df['name_luxury_count'], df['price'], alpha=0.4, s=15, color='darkblue')
    axes[0, 1].set_xlabel('Luxury Keywords Count', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Price ($)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Luxury Keywords vs Price', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim(0, 500)
    axes[0, 1].grid(alpha=0.3)

    # 3. Budget words vs price
    axes[1, 0].scatter(df['name_budget_count'], df['price'], alpha=0.4, s=15, color='orange')
    axes[1, 0].set_xlabel('Budget Keywords Count', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Price ($)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Budget Keywords vs Price', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim(0, 500)
    axes[1, 0].grid(alpha=0.3)

    # 4. Exclamation mark effect
    exclaim_data = df.groupby('name_has_exclamation')['price'].agg(['mean', 'median', 'count'])
    categories = ['No Exclamation', 'Has Exclamation']
    axes[1, 1].bar(categories, exclaim_data['median'].values, color=['coral', 'teal'], alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Median Price ($)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Effect of Exclamation Mark on Price', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)

    for i, v in enumerate(exclaim_data['median'].values):
        axes[1, 1].text(i, v + 5, f'${v:.0f}\n(n={exclaim_data["count"].values[i]})',
                       ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('NLP Features Analysis', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '10_nlp_features_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: 10_nlp_features_analysis.png")
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Run the complete analysis pipeline.
    """
    print("\n")
    print("="*80)
    print("NYC AIRBNB PRICING PREDICTION - ANALYSIS PIPELINE")
    print("="*80)
    print(f"QTM 347 Machine Learning Course Project - Emory University")
    print(f"Team: Connor Lee, Kaya Monrose, Parker Shimp, Sam Besley")
    print("="*80)

    # 1. Load data
    print("\n[STEP 1] Loading data...")
    df = load_data(DATA_DIR / "AB_NYC_2019.csv")
    display_basic_stats(df)

    # 2. Clean data
    print("\n[STEP 2] Cleaning data...")
    df_clean = clean_data(df)

    # 3. Engineer NLP features
    print("\n[STEP 3] Engineering NLP features from listing names...")
    df_nlp = engineer_nlp_features(df_clean)

    # 4. Engineer additional features
    print("\n[STEP 4] Engineering features...")
    df_feat, numeric_features, categorical_features = engineer_features(df_nlp)

    # 5. Prepare modeling data
    print("\n[STEP 5] Preparing data for modeling...")
    X_train, X_test, y_train, y_test, feature_names = prepare_modeling_data(
        df_feat, numeric_features, categorical_features
    )

    # 6. Train models
    print("\n[STEP 6] Training models...")
    results = train_models(X_train, X_test, y_train, y_test, feature_names)

    # 7. Generate visualizations
    print("\n[STEP 7] Generating visualizations...")
    print("\nCreating figures...")
    plot_price_distributions(df_clean)
    plot_price_by_borough(df_clean)
    plot_price_by_room_type(df_clean)
    plot_geographic_scatter(df_clean)
    plot_top_neighborhoods(df_clean)
    plot_correlation_heatmap(df_feat, numeric_features)
    plot_model_comparison(results)
    plot_predicted_vs_actual(results)
    plot_feature_importance(results)
    plot_nlp_features_analysis(df_feat)

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print("\nGenerated figures:")
    print("  1. 01_price_distributions.png - Raw and log-transformed price distributions")
    print("  2. 02_price_by_borough.png - Price distribution by NYC borough")
    print("  3. 03_price_by_room_type.png - Price distribution by room type")
    print("  4. 04_geographic_scatter.png - Geographic distribution of listings")
    print("  5. 05_top_neighborhoods.png - Top 15 neighborhoods by median price")
    print("  6. 06_correlation_heatmap.png - Feature correlation matrix")
    print("  7. 07_model_comparison.png - Model performance comparison (RMSE, R², MAE)")
    print("  8. 08_predicted_vs_actual.png - Predicted vs actual prices for all models")
    print("  9. 09_feature_importance.png - Top 15 most important features")
    print("  10. 10_nlp_features_analysis.png - NLP features analysis")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
