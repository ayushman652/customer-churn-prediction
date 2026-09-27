
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.trainer import train_model
from src.evaluator import evaluate_model
from src.visualizer import plot_coefficients


def main():
    df = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    plot_coefficients(model)


if __name__ == "__main__":
    main()