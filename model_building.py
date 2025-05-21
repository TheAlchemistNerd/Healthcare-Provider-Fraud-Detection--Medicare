from sklearn.metrics import (accuracy_score, classification_report, 
                             RocCurveDisplay, confusion_matrix, 
                             ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
import matplotlib.pyplot as plt


# Define functions for model fitting, tuning, and evaluation
def naive_model(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)
    return classifier

def tuned_model(classifier, param_grid, X_train, y_train):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = GridSearchCV(classifier, param_grid, cv=cv, scoring='accuracy')
    grid.fit(X_train, y_train)
    return grid

def evaluate_model(classifier, X_valid, y_valid):
    # Check if classifier is an instance of GridSearchCV
    if isinstance(classifier, GridSearchCV):
        best_classifier = classifier.best_estimator_
    else:
        best_classifier = classifier
        
    # Predict on validation set
    y_pred = best_classifier.predict(X_valid)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    
    # Print accuracy and classification report
    print(f"Accuracy: {accuracy*100:.3f}%")
    print("Classification Report:\n", classification_report(y_valid, y_pred))
    
    RocCurveDisplay.from_predictions(y_valid, y_pred)
    
    # Display confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    display(cm)
    ConfusionMatrixDisplay.from_predictions(y_valid, y_pred)
    
    plt.tight_layout()
    plt.show()