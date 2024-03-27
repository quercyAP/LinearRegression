# Car Price Prediction

This project implements a simple linear regression model to predict the price of a car based on its mileage. It's designed to demonstrate the fundamentals of machine learning, specifically linear regression, and how it can be applied to real-world data.

## Installation
Ensure you have Python 3.x installed on your machine.
1. **Clone the Repository**:

```
git clone https://github.com/quercyAP/LinearRegression
cd LinearRegression
```

2. **Install Dependencies**:

```
pip install numpy matplotlib
```

## Usage

### Training the Model

The model is trained using a simple linear regression algorithm. The `train.py` script reads data from a `data.csv` file, trains the linear regression model on this data, and saves the model's parameters (`theta0` and `theta1`) in `model_params.txt`.

Linear regression works by adjusting parameters (theta0 and theta1) to minimize the cost function, which in this case is the mean squared error (MSE) between the predicted prices and the actual prices in the dataset. The training process uses gradient descent to iteratively adjust these parameters.

To train the model, execute:
```
python train.py
```
This will also plot the MSE for each iteration and the regression line, demonstrating the model's fit.

Predicting Car Prices
With the model trained, use predict.py to estimate car prices based on mileage:
```
python predict.py
```
This script normalizes input mileage for consistency with the training process and uses the model parameters for prediction.

### Algorithm Details

Training Algorithm

- Normalization: Mileages are normalized using feature scaling.
- Gradient Descent: Updates theta0 and theta1 by subtracting the gradient of the cost function, scaled by the learning rate.
- MSE Calculation: Monitors model performance at each iteration.
  
Prediction Algorithm
- Normalization: User's mileage input is normalized using the training dataset's mean and standard deviation.
- Price Estimation: Applies linear regression formula with normalized mileage for price estimation.
