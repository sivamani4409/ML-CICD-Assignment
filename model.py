
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model():
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)


def main():
    model, X_test, y_test = train_model()
    accuracy = evaluate_model(model, X_test, y_test)

    accuracy_percent = accuracy * 100

    print(f"Model: Logistic Regression")
    print(f"Dataset: Iris")
    print(f"Model Accuracy: {accuracy_percent:.2f}%")

    with open("accuracy.txt", "w") as file:
        file.write(f"{accuracy_percent:.2f}%")


if __name__ == "__main__":
    main()
