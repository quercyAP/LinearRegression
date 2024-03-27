import numpy as np
import matplotlib.pyplot as plt

def plot_data_and_regression_line(X, Y, theta0, theta1, mean_X, std_X):
    plt.figure() 
    plt.scatter(X, Y, color='blue', label='Données réelles')
    
    X_line = np.linspace(min(X), max(X), 100)
    Y_line = theta0 + theta1 * ((X_line - mean_X) / std_X) 
    
    plt.plot(X_line, Y_line, color='red', label='Ligne de régression')
    plt.title('Données et ligne de régression linéaire')
    plt.xlabel('Kilométrage')
    plt.ylabel('Prix')
    plt.legend()
    plt.show(block=False)

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data[:, 0], data[:, 1]

def normalize_features(X):
    mean = np.mean(X)
    std = np.std(X)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def train_linear_regression(X, Y, learning_rate, iterations):
    m = len(X)
    theta0 = 0
    theta1 = 0
    mse_history = [] 
    
    for _ in range(iterations):
        sum_errors_theta0 = 0
        sum_errors_theta1 = 0
        for i in range(m):
            prediction = estimate_price(X[i], theta0, theta1)
            error = prediction - Y[i]
            sum_errors_theta0 += error
            sum_errors_theta1 += error * X[i]
        
        theta0 -= (learning_rate * (1/m) * sum_errors_theta0)
        theta1 -= (learning_rate * (1/m) * sum_errors_theta1)

        mse = calculate_mse(X, Y, theta0, theta1)
        mse_history.append(mse)
    
    return theta0, theta1, mse_history

def save_model_parameters(theta0, theta1, mean_X, std_X, filename="model_params.txt"):
    with open(filename, 'w') as f:
        f.write(f"{theta0}\n{theta1}\n{mean_X}\n{std_X}")

def calculate_mse(X, Y, theta0, theta1):
    m = len(X)
    total_error = 0.0
    for i in range(m):
        prediction = estimate_price(X[i], theta0, theta1)
        error = prediction - Y[i]
        total_error += (error ** 2)
    mse = total_error / m
    return mse


def main():
    X, Y = load_data('data.csv')
    X_normalized, mean_X, std_X = normalize_features(X)
    
    learning_rate = 0.01
    iterations = 1000
    theta0, theta1, mse_history = train_linear_regression(X_normalized, Y, learning_rate, iterations)
    
    save_model_parameters(theta0, theta1, mean_X, std_X)

    print(f"Modèle entraîné : theta0 = {theta0}, theta1 = {theta1}")
    print(f"Données normalisées : mean = {mean_X}, std = {std_X}")
    
    # Calculer les prédictions sur les données normalisées
    Y_predicted = [estimate_price(x, theta0, theta1) for x in X_normalized]

    # Calculer et afficher MSE
    mse = calculate_mse(X_normalized, Y, theta0, theta1)
    print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")

    plot_data_and_regression_line(X, Y, theta0, theta1, mean_X, std_X)

    # Tracer la MSE
    plt.figure() 
    plt.plot(mse_history)
    plt.title('Evolution de la MSE par itération')
    plt.xlabel('Itération')
    plt.ylabel('MSE')
    plt.show()

if __name__ == "__main__":
    main()
