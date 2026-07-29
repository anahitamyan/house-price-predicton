# sklearn_predictor.py
from sklearn.linear_model import LinearRegression

def main():
    # Original 1D datasets
    house_sizes_sqm = [50, 70, 90, 120, 150]
    house_prices_thousands = [100, 150, 200, 280, 350]

    # Reshape features (X) into a 2D list of lists: [[50], [70], [90], ...]
    reshaped_sizes = [[size] for size in house_sizes_sqm]

    # Initialize and train the model
    model = LinearRegression()
    model.fit(reshaped_sizes, house_prices_thousands)

    # Print learned parameters (weight/slope and bias/intercept)
    print(f"Model trained successfully!")
    print(f"  - Weight (Slope):     {model.coef_[0]:.4f} k$/m²")
    print(f"  - Bias (Intercept):   {model.intercept_:.4f} k$\n")

    # Predict price for a 100 m² house
    new_house_size = [[200]]
    predicted_price = model.predict(new_house_size)[0]

    print(f"Predicted price for a {new_house_size[0][0]} m² house:")
    print(f"  -> {predicted_price:.2f} k$ (${predicted_price * 1000:,.2f} USD)")

if __name__ == "__main__":
    main()