# single_variable_predictor.py

class SingleVariablePredictor:
    def __init__(self):
        self.w = 0.0  # Weight (slope)
        self.b = 0.0  # Bias (intercept)
        self.is_trained = False

    def fit(self, house_sizes_sqm, house_prices_thousands):
        """Train the model parameters (w and b) using Ordinary Least Squares."""
        n = len(house_sizes_sqm)
        if n == 0 or len(house_prices_thousands) != n:
            raise ValueError("Sizes and prices must be non-empty and equal in length.")

        mean_x = sum(house_sizes_sqm) / n
        mean_y = sum(house_prices_thousands) / n

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(house_sizes_sqm, house_prices_thousands))
        denominator = sum((x - mean_x) ** 2 for x in house_sizes_sqm)

        self.w = numerator / denominator
        self.b = mean_y - (self.w * mean_x)
        self.is_trained = True

    def predict(self, size):
        """Predict price for a single house size."""

        return self.w * size + self.b

    def print_predictions(self, house_sizes_sqm, house_prices_thousands):
        """Compare predictions against actual prices for a dataset."""

        print("\n--- Training Set Predictions ---")
        for size, actual in zip(house_sizes_sqm, house_prices_thousands):
            y_pred = self.predict(size)  # Uses self.w and self.b internally
            print(f"House size: {size:>3} m² | Predicted: {y_pred:6.1f} k$ | Actual: {actual:>3} k$")
        print("--------------------------------\n")

if __name__ == "__main__":
    house_sizes_sqm = [50, 70, 90, 120, 150]
    house_prices_thousands = [100, 150, 200, 280, 350]

    model = SingleVariablePredictor()
    model.fit(house_sizes_sqm, house_prices_thousands)
    model.print_predictions(house_sizes_sqm, house_prices_thousands)

    print("--- Real Estate Price Predictor ---")
    try:
        user_input = float(input("Enter house size in m² (e.g., 90): "))
        predicted_price = model.predict(user_input)
        print(f"\nEstimated Listing Price for a {user_input:.1f} m² house:")
        print(f"  -> ${predicted_price * 1000:,.2f} USD")
    except ValueError:
        print("Invalid input. Please enter a numerical value.")