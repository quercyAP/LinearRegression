import numpy as np

def load_model_parameters(filename="model_params.txt"):
    with open(filename, 'r') as f:
        theta0 = float(f.readline())
        theta1 = float(f.readline())
        mean_X = float(f.readline())
        std_X = float(f.readline())
    return theta0, theta1, mean_X, std_X


def normalize_input(input_value, mean, std):
    return (input_value - mean) / std

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def main():
    theta0, theta1, mean_X, std_X = load_model_parameters()
    
    user_input = float(input("Entrez le kilométrage de la voiture : "))
    normalized_input = normalize_input(user_input, mean_X, std_X)
    
    estimated_price = estimate_price(normalized_input, theta0, theta1)
    print(f"Le prix estimé pour une voiture avec {int(user_input)} km est de {estimated_price:.2f} euros.")

if __name__ == "__main__":
    main()
