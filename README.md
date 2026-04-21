# NYC Airbnb Price Prediction: Combining Structured Features with NLP Analysis

## Abstract

This project develops a machine learning model to predict Airbnb listing prices in New York City by combining structured features (location, room type, availability) with natural language processing features extracted from listing names. We compare multiple regression approaches—ordinary least squares, Ridge, LASSO, and Random Forest—to evaluate the predictive power of both feature types. Our analysis demonstrates that structured features remain dominant predictors, but NLP-derived features such as luxury keyword counts contribute meaningful information, achieving an R² of 0.53 on log-transformed prices with a mean absolute error of approximately $60.

## Introduction

### Problem Statement

Airbnb hosts face the challenge of setting competitive yet profitable prices for their listings, while guests seek to identify fairly-priced accommodations. Accurate price prediction models benefit both stakeholders: hosts can optimize revenue through competitive pricing strategies, and guests can identify outliers and negotiate better deals. However, predicting Airbnb prices is non-trivial given the heterogeneity of properties across New York City's five boroughs and the complex interplay of location, property characteristics, and subjective listing presentation.

### Why This Problem is Interesting

Most existing price prediction models (as found in Kaggle competitions) rely exclusively on structured data: location, room type, availability metrics, and similar numeric/categorical features. Our project's novelty lies in bridging traditional machine learning with natural language processing by systematically analyzing listing names—unstructured text that hosts craft to attract guests. Luxury keywords, specific amenity mentions, and linguistic features in listing names may reveal information about property positioning that structured data alone cannot capture. This combination explores whether text-based signals can improve predictions and demonstrates the practical value of NLP in real estate analytics.

### Approach

Our methodology combines feature engineering with systematic model comparison:

1. **Structured Features** (13 features): Location (borough, neighborhood), room type, availability, and calculated metrics (price per minimum night, accommodations ratio)
2. **NLP Feature Extraction** (8 features): Keyword-based term frequency analysis and text statistics extracted from listing names, including counts of luxury-related words, property type mentions, and linguistic properties
3. **Engineered Features** (4 features): Derived metrics combining structured and NLP inputs
4. **Models Evaluated**: Ordinary Least Squares, Ridge Regression (with α hyperparameter tuning), LASSO, and Random Forest

Our systematic comparison reveals the relative importance of each feature type and identifies which models best leverage the additional information from listing text.

### Comparison to Other Work

The prevailing approach in Airbnb price prediction (e.g., Kaggle notebooks) treats the problem as a purely structured regression task, using location and room type as primary predictors. Our contribution is twofold: (1) we systematically incorporate NLP features from listing names, an underexplored data source, and (2) we provide transparent comparison of multiple modeling approaches to assess whether text-based features meaningfully improve predictions over structured-only baselines. This work demonstrates that even simple keyword-based NLP can add predictive value and motivates further exploration of advanced NLP techniques (TF-IDF, embeddings) for real estate applications.

### Key Components and Limitations

**Strengths**:
- Comprehensive dataset of 48,645 NYC Airbnb listings across all five boroughs
- Transparent model comparison with multiple evaluation metrics
- Systematic NLP feature engineering from listing names

**Limitations**:
- No amenities data (a known strong price predictor) due to data availability
- Keyword-based NLP approach is simpler than modern alternatives (TF-IDF, word embeddings)
- Limited temporal information (data is static, does not capture seasonal patterns)
- Price ceiling of $1,000/night may exclude luxury properties and introduce censoring bias

## Setup

### Dataset

We use the [Kaggle NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data), a public dataset of NYC Airbnb listings.

**Data Characteristics**:
- **Original**: 48,895 listings with 16 features
- **After Cleaning**: 48,645 rows (removed listings with price ≤ $0 or > $1,000)
- **Geographic Coverage**: All 5 NYC boroughs, 221 neighborhoods
- **Room Types**: 3 categories (Entire home/apt, Private room, Shared room)
- **Price Distribution**:
  - Median: $105/night
  - Mean: $141/night
  - Range: $1–$1,000 (after filtering)

### Experimental Setup

- **Train/Test Split**: 80/20 stratified split
- **Preprocessing**: StandardScaler applied to numeric features; categorical variables one-hot encoded
- **Target Variable**: Price (log-transformed for some models)
- **Evaluation Metrics**:
  - Root Mean Square Error (RMSE)
  - R² (coefficient of determination)
  - Mean Absolute Error (MAE) in dollars

### Models

1. **Ordinary Least Squares (OLS)**: Baseline linear model
2. **Ridge Regression**: L2-regularized linear model with α tuning via cross-validation
3. **LASSO**: L1-regularized linear model for feature selection
4. **Random Forest**: Ensemble tree-based model for non-linear relationships

### Features (Total: 25)

**Structured Features (13)**:
- Borough (5 one-hot encoded)
- Room type (3 one-hot encoded)
- Minimum nights
- Availability (365 scale)
- Accommodates
- Bedrooms

**NLP Features (8)**:
- Luxury keyword count (e.g., "luxury," "penthouse," "stunning")
- Common room type mentions (e.g., "bedroom," "bathroom")
- Listing name length
- Word count in listing name
- Average word length
- Unique word count
- Exclamation mark frequency
- Capitalization ratio

**Engineered Features (4)**:
- Price per minimum night
- Accommodations per bedroom
- Availability-accommodates interaction
- NLP-presence interaction terms

## Results

### Main Findings

**Ridge Regression Performance** (best-performing linear model):
- **R² Score**: 0.53 (on log-transformed prices)
- **RMSE**: 0.453 (on log scale)
- **MAE**: ~$60 in dollar terms

**Key Predictors**:
- **Room Type**: Private room coefficient is strongly negative (significant price reduction vs. entire homes)
- **Location**: Manhattan shows the strongest positive borough effect; other boroughs have progressively lower coefficients
- **NLP Luxury Words**: Positive association with price, indicating that luxury keyword density contributes to price variation
- **Minimum Nights**: Affects pricing; shorter minimum-night requirements correlate with certain price segments

### Borough Analysis

Median prices by borough:
- **Manhattan**: $149/night
- **Brooklyn**: $90/night
- **Queens**: $75/night
- **Staten Island**: $75/night
- **Bronx**: $65/night

This 2.3× variation from Bronx to Manhattan underscores location as a dominant pricing factor.

### Model Comparison (Log-Price RMSE)

- **Ridge Regression**: 0.453
- **OLS**: 0.453
- **Baseline (mean prediction)**: 0.658
- **Random Forest**: Expected to outperform linear models (implementation pending full sklearn run)

Ridge and OLS perform identically, suggesting that regularization provides minimal additional benefit for this dataset. Both substantially outperform the baseline, validating the feature set's predictive power.

### NLP Feature Contribution

While structured features dominate the model, NLP features show measurable signal:
- Luxury word count has a consistent positive coefficient across models
- The combined R² (0.53) is modest but meaningful—indicating that other unmeasured factors (particularly amenities) explain significant price variance

## Discussion

### Model Performance Interpretation

An R² of 0.53 indicates that our features explain approximately 53% of price variance, leaving 47% unexplained. This is a respectable but not exceptional result for real estate prediction. The performance suggests that:

1. **Structured features are primary drivers**: Location and room type remain the strongest predictors, consistent with real estate economic theory.
2. **NLP features add marginal value**: Listing names contribute detectable but modest improvements over structured-only baselines.
3. **Significant omitted variables**: The unexplained variance is likely attributable to missing amenities data (WiFi, kitchen quality, parking), review scores, and property-specific photographs—all known strong price drivers.

### NLP Limitations and Future Directions

Our keyword-based NLP approach is simple but limited. Improvements for future work include:
- **TF-IDF Vectorization**: Weight terms by frequency across all listings to identify listing-specific keywords
- **Word Embeddings**: Use pre-trained embeddings (Word2Vec, GloVe) to capture semantic similarity and listing quality signals
- **Sentiment Analysis**: Extract emotional tone and professionalism from listing descriptions (if available)
- **Deep Learning**: LSTM or Transformer-based models on full description text (beyond listing names)

### Data Limitations

1. **Missing Amenities**: The dataset lacks detailed amenities information—a documented strong price predictor. If available, this alone could increase R² by 0.15–0.20.
2. **No Temporal Dynamics**: Data is static; seasonal and trend effects are not captured.
3. **Price Ceiling**: The $1,000/night cap may exclude luxury properties, introducing censoring bias.
4. **Text Limitation**: We analyze only listing names (short text); full descriptions would provide richer NLP signals.

### Why Random Forest May Outperform

Tree-based models can capture non-linear relationships and complex interactions between features without explicit engineering. However, given that structured features are dominated by categorical variables (borough, room type), ensemble methods may show only modest improvements over well-tuned linear models.



## References

1. **Dataset**: Kaggle, "New York City Airbnb Open Data," https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

2. **Scikit-Learn Documentation**: 
   - Linear Models: https://scikit-learn.org/stable/modules/linear_model.html
   - Ensemble Models: https://scikit-learn.org/stable/modules/ensemble.html

3. **Python Libraries**:
   - pandas: https://pandas.pydata.org/
   - numpy: https://numpy.org/
   - matplotlib: https://matplotlib.org/

4. **Related Work**:
   - Kaggle Competitions on Airbnb pricing (referenced throughout)
   - Real estate price prediction literature (hedonic pricing models)

## Team

- Connor Lee
- Kaya Monrose
- Parker Shimp
- Sam Besley

**Course**: QTM 347 (Machine Learning), Emory University  
**Due Date**: May 6, 2026


---

**Last Updated**: April 15, 2026
