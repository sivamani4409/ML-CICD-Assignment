
from model import train_model, evaluate_model


def test_model_trains_successfully():
    model, X_test, y_test = train_model()

    assert model is not None
    assert len(X_test) > 0
    assert len(y_test) > 0


def test_model_predictions():
    model, X_test, y_test = train_model()

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_model_accuracy():
    model, X_test, y_test = train_model()

    accuracy = evaluate_model(model, X_test, y_test)

    assert accuracy >= 0.90
    assert accuracy <= 1.0
