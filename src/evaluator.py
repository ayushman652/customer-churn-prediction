from sklearn.metrics import log_loss

def evaluate_model(model, x_test, y_test):
    y_pred  = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    
    print("Predicted classes: ", y_pred)
    print("\nPredicted probabilities:\n", y_prob)
    print("\nLog Loss: ", log_loss(y_test, y_prob))