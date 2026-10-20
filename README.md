# California Housing Price Prediction

## Project Overview

This project uses machine learning to predict California house values based on housing and demographic features. It explores data analysis, preprocessing, model training, and model evaluation using Python.

## Objectives

* Explore and visualize the California Housing dataset.
* Analyze relationships between housing features.
* Split the data into training and testing sets.
* Train a Linear Regression model.
* Evaluate model performance using RMSE and MAE.
* Use cross-validation and a machine learning pipeline.
* Tune a Random Forest model using Grid Search and Randomized Search.

## Technologies Used

* Python
* pandas
* scikit-learn
* Matplotlib
* Seaborn

## Dataset

The project uses the California Housing dataset provided by scikit-learn. It contains housing-related features and the target variable `MedHouseVal`.

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Perform exploratory data analysis and visualizations.
3. Split the data into training and testing sets.
4. Scale features for Linear Regression.
5. Train and evaluate the Linear Regression model.
6. Perform cross-validation.
7. Tune Random Forest hyperparameters.
8. Evaluate the selected model on the test set.

## Evaluation Metrics

* **RMSE (Root Mean Squared Error):** Measures prediction error, penalizing larger errors more heavily.
* **MAE (Mean Absolute Error):** Measures the average absolute difference between predicted and actual values.

Lower values generally indicate better predictions.

## How to Run

Clone this repository and navigate into the project directory:

```bash
git clone https://github.com/Megnkemgit/california-housing-ml.git
cd california-housing-ml
```

Install the required libraries:

```bash
python -m pip install pandas scikit-learn matplotlib seaborn
```

Run the Python script:

```bash
python california.py
```

## Future Improvements

* Compare additional regression models.
* Visualize actual versus predicted house values.
* Document final model metrics and tuning results.
* Save the trained model for future predictions.

## Author

Maggie Emefiele
