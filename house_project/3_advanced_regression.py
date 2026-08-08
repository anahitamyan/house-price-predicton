# 3_advanced_regression.py
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def demonstrate_feature_scaling():

    X = [
        [1500, 2],
        [1300, 10],
        [2000, 5],
        [1200, 20]
    ]
    y = [300, 250, 400, 200]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nRaw Features (X):")
    for row in X:
        print(f"  Size: {row[0]:>4} sq ft | Age: {row[1]:>2} years")

    print("\nScaled Features (X_scaled) - Z-score normalized:")
    for row in X_scaled:
        print(f"  Size: {row[0]:>6.2f}       | Age: {row[1]:>6.2f}")

    print("\nNotice how all numbers are now centered around 0 (mostly between -2 and +2)!")

    model = LinearRegression()
    model.fit(X_scaled, y)

    print("\nModel Trained on Scaled Data:")
    print(f"  - Weight for Scaled Size: {model.coef_[0]:.2f} k$")
    print(f"  - Weight for Scaled Age:  {model.coef_[1]:.2f} k$")
    print(f"  - Intercept (Bias):       {model.intercept_:.2f} k$\n")


if __name__ == "__main__":
    demonstrate_feature_scaling()