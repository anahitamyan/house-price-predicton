# 1_mse_calculator.py
# MSE (Mean Squared Error) measures how far off your model's predictions are from the actual values.
def calculate_mse(actual_y, predicted_y):
    # Formula: MSE (Mean Squared Error )= (1 / N) * sum((actual - predicted)^2) 
    n = len(actual_y)
    if n == 0 or len(predicted_y) != n:
        raise ValueError("Actual and predicted lists must be non-empty and of equal length.")

    squared_errors = [(actual - pred) ** 2 for actual, pred in zip(actual_y, predicted_y)]
    mse = sum(squared_errors) / n
    return mse


def make_predictions(sizes, weight, bias):
    #  y_hat = w * x + b for all sizes.
    return [(weight * size) + bias for size in sizes]


if __name__ == "__main__":
    # The Dataset
    sizes = [50, 70, 90, 120, 150]
    actual_prices = [100, 150, 200, 280, 350]

    print("=== Task 2 Part 1: MSE Calculator ===")
    print("Dataset Sizes (m²):", sizes)
    print("Actual Prices (k$):", actual_prices)
    print("-" * 50)

    # Pre-calculated test runs from the assignment
    # bad_preds = make_predictions(sizes, 1.0, 0.0)
    # better_preds = make_predictions(sizes, 2.3, 0.0)
    # print(f"Reference Guess 1 (w=1.0, b=0.0) -> MSE: {calculate_mse(actual_prices, bad_preds):.2f}")
    # print(f"Reference Guess 2 (w=2.3, b=0.0) -> MSE: {calculate_mse(actual_prices, better_preds):.2f}")
    # print("-" * 50)

    while True:
        try:
            
            user_w_input = input("  Enter Weight (e.g., 2.3): ").strip()
            if user_w_input.lower() == 'q': # for quit the program
                print("Exiting MSE Calculator!")
                break
            
            user_b_input = input("  Enter Bias (e.g., 10.0): ").strip()
            if user_b_input.lower() == 'q': # for quit the program
                print("Exiting MSE Calculator!")
                break

            weight = float(user_w_input)
            bias = float(user_b_input)

            predictions = make_predictions(sizes, weight, bias)
            mse = calculate_mse(actual_prices, predictions)

            print(f"\nResults for Weight = {weight}, Bias = {bias}:")
            for size, actual, pred in zip(sizes, actual_prices, predictions): # zip is used to iterate over the sizes, actual_prices, and predictions
                print(f"  Size: {size:>3} m² | Actual: {actual:>3} k$ | Predicted: {pred:6.1f} k$ | Error: {actual - pred:6.1f}")
            
            print(f"  -> Calculated MSE: {mse:.2f}")

        except ValueError:
            print("❌ Invalid input. Please enter valid numerical values.")