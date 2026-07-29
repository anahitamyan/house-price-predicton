# sklearn_two_variable_predictor.py
from sklearn.linear_model import LinearRegression

def main():
    # Dataset: [Size (sqm), Number of Bedrooms]
    house_features = [
        [50, 1],
        [70, 2],
        [90, 2],
        [120, 3],
        [150, 4]
    ]
    house_prices_thousands = [100, 150, 200, 280, 350]

    model = LinearRegression()
    model.fit(house_features, house_prices_thousands)

    print("Model trained successfully!")
    print(f"  - Weight for Size (w1):    {model.coef_[0]:.4f} k$/m²")
    print(f"  - Weight for Bedrooms (w2): {model.coef_[1]:.4f} k$/bedroom")
    print(f"  - Bias / Intercept (b):     {model.intercept_:.4f} k$\n")

    print("--- Real Estate Price Predictor (Scikit-Learn) ---")
    try:
        user_size = float(input("Enter house size in m² (e.g., 100): "))
        user_bedrooms = float(input("Enter number of bedrooms (e.g., 2): "))

        new_house = [[user_size, user_bedrooms]]
        predicted_price = model.predict(new_house)[0]

        print(f"\nEstimated Listing Price for a {user_size:.1f} m² house with {int(user_bedrooms)} bed(s):")
        print(f"  ->${predicted_price * 1000:,.2f} USD)")

    except ValueError:
        print("Invalid input. Please enter valid numbers.")

if __name__ == "__main__":
    main()