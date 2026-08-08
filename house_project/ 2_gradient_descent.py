# 2_gradient_descent.py

def calculate_mse(actual_y, predicted_y):
    # MSE (Mean Squared Error )= (1 / N) * sum((actual - predicted)^2) 
    n = len(actual_y)
    squared_errors = [(actual - pred) ** 2 for actual, pred in zip(actual_y, predicted_y)]
    return sum(squared_errors) / n


def run_gradient_descent():
    sizes = [50, 70, 90, 120, 150]
    actual_prices = [100, 150, 200, 280, 350]
    n = len(sizes)

    weight = 0.0
    bias = 0.0
    learning_rate = 0.00001  #if the learning rate is too high, the model will diverge
    epochs = 100

    # print(f"Initial Setup: Weight={weight}, Bias={bias}, Learning Rate={learning_rate}\n")

    for epoch in range(1, epochs + 1):
        # y_hat = weight * x + bias
        predictions = [(weight * x) + bias for x in sizes]

        # weight_derivative = -2/N * sum( x * (actual_y - predicted_y) )
        weight_derivative = (-2 / n) * sum(x * (actual - pred) for x, actual, pred in zip(sizes, actual_prices, predictions))
        
        # bias_derivative = -2/N * sum( actual_y - predicted_y )
        bias_derivative = (-2 / n) * sum(actual - pred for actual, pred in zip(actual_prices, predictions))

        weight -= learning_rate * weight_derivative
        bias -= learning_rate * bias_derivative

        current_mse = calculate_mse(actual_prices, predictions)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}: MSE = {current_mse:>10.2f} | Weight = {weight:.4f} | Bias = {bias:.4f}")

    print("\nFinal Model Parameters:")
    print(f"Learned Weight: {weight:.4f}")
    print(f"Learned Bias:   {bias:.4f}")
    
    final_preds = [(weight * x) + bias for x in sizes]
    final_mse = calculate_mse(actual_prices, final_preds)
    print(f"Final Trained MSE: {final_mse:.2f}")


if __name__ == "__main__":
    run_gradient_descent()