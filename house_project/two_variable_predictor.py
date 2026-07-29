# two_variable_predictor.py

class TwoVariablePredictor:
    def __init__(self):
        self.w1 = 2.0  # Weight (for size)
        self.w2 = 10.0  # Weight (for bedrooms)
        self.b = 10.0  # Bias (intercept)
        self.is_trained = False


    def predict(self, size, bedrooms):
        """Predict price for a single house size."""

        return (self.w1 * size) + (self.w2 * bedrooms) + self.b

    def print_predictions(self, house_features, house_prices_thousands):
        """Compare predictions against actual prices for a dataset."""

        print("\n--- Training Set Predictions ---")
        for house, actual in zip(house_features, house_prices_thousands):
            size = house[0]
            bedrooms = house[1]
            y_pred = self.predict(size, bedrooms)
            print(f"Size: {size:>3} m² | Bedrooms: {bedrooms} | Predicted: {y_pred:6.1f} k$ | Actual: {actual:>3} k$")
        print("----------------------------------------------------\n")

if __name__ == "__main__":
    house_features = [
        [50, 1],
        [70, 2],
        [90, 2],
        [120, 3],
        [150, 4]
    ]
    house_prices_thousands = [100, 150, 200, 280, 350]

    model = TwoVariablePredictor()
    model.print_predictions(house_features, house_prices_thousands)

    print("--- Real Estate Price Predictor ---")
    try:
        user_size = float(input("Enter house size in m² (e.g., 100): "))
        user_bedrooms = float(input("Enter number of bedrooms (e.g., 3): "))

        predicted_price = model.predict(user_size, user_bedrooms)

        print(f"\nEstimated Listing Price for a {user_size:.1f} m² house with {int(user_bedrooms)} bed(s):")
        print(f"  -> ${predicted_price * 1000:,.2f} USD")
    except ValueError:
        print("Invalid input. Please enter a numerical value.")