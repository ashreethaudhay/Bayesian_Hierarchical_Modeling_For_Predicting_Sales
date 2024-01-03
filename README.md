# Bayesian Hierarchical Modeling for Predicting Sales

## Overview
This project applies Bayesian hierarchical modeling techniques to predict product sales at stores in Ecuador using historical transaction data. A variety of features are engineered from the raw data and several models are evaluated, with a focus on accounting for the nested structure of the data across hierarchical levels like stores, regions, and item categories.

## Data
The following data sets are included:
- train.csv - Historical training data with the target unit_sales variable to predict
- test.csv - Test set to predict sales on
- items.csv - Metadata on products
- stores.csv - Metadata on stores
- transactions.csv - Historical transaction data
- holidays_events.csv - Information on holidays and events
- oil.csv - Ecuador oil price data

## Preprocessing
The Python script performs several preprocessing steps:
- Handle missing values
- Encode categorical variables
- Feature engineering from date columns
- Merge additional data sets like oil prices and holidays
- Add weights to perishable items

## Modeling
A Bayesian Ridge regression model is fit on the training data to predict unit_sales. Cross-validation is used to evaluate performance.
The script is structured to easily add additional models to the apply_models function for comparison.

## Results
Model scores on 2-fold cross-validation are output, along with a bar plot comparing performance across models.

## Next Steps
Potential next steps include:
- Adding hierarchical structure to the model to account for store and product hierarchies
- Testing additional model types like regularized regression, random forests, or neural networks
- Performing hyperparameter optimization for the best model
- Stacking models into an ensemble
