from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, roc_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import shap
from sklearn.tree import plot_tree
import mpld3
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("Data.csv")
def NB(X_train, y_train, X_test, y_test):
    """
    Perform Gaussian Naive Bayes classification on the given training and test data.

    Parameters:
        X_train (numpy.ndarray or pandas.DataFrame): The training data features.
        y_train (numpy.ndarray or pandas.Series): The training data labels.
        X_test (numpy.ndarray or pandas.DataFrame): The test data features.
        y_test (numpy.ndarray or pandas.Series): The test data labels.

    Returns:
        None

    Example:
        NB(X_train, y_train, X_test, y_test)
    """
    NB = GaussianNB()
    NB.fit(X_train, y_train)
    NB_pred = NB.predict(X_test)
    print(f"Accuracy score for GaussianNB is {accuracy_score(y_test, NB_pred)}")

    # Let's plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(
        y_test, NB_pred), display_labels=['Class 0', 'Class 1'])

    disp.plot(cmap=plt.cm.Blues)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Let's plot the ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, NB_pred)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score(
        y_test, NB_pred), estimator_name='Gausian Naive Bayes')
    roc_display.plot()
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()


def ada(X_train, y_train, X_test, y_test):
    """
    Perform AdaBoost classification on the given training and test data.

    Parameters:
        X_train (numpy.ndarray or pandas.DataFrame): The training data features.
        y_train (numpy.ndarray or pandas.Series): The training data labels.
        X_test (numpy.ndarray or pandas.DataFrame): The test data features.
        y_test (numpy.ndarray or pandas.Series): The test data labels.

    Returns:
        None

    Example:
        ada(X_train, y_train, X_test, y_test)
    """
    ada_classifier = AdaBoostClassifier()
    param_grid = {
        'n_estimators': [30, 31, 32, 33, 34, 35],  # Number of base estimators
        'learning_rate': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Learning rate
    }
    random_search = RandomizedSearchCV(
        ada_classifier, param_distributions=param_grid, n_iter=10, cv=5)
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("ٍِِِada hyperparams:", best_params)
    print("ada Best Score:", best_score)

def RF(X_train, y_train, X_test, y_test):
    """
    Perform Random Forest classification on the given training and test data.

    Parameters:
        X_train (numpy.ndarray or pandas.DataFrame): The training data features.
        y_train (numpy.ndarray or pandas.Series): The training data labels.
        X_test (numpy.ndarray or pandas.DataFrame): The test data features.
        y_test (numpy.ndarray or pandas.Series): The test data labels.

    Returns:
        None

    Example:
        RF(X_train, y_train, X_test, y_test)
    """
    # Let's build our model
    rf_classifier = RandomForestClassifier(n_estimators=300, random_state=42)
    rf_classifier.fit(X_train, y_train)
    rf_y_pred = rf_classifier.predict(X_test)

    print("RF Accuracy:", accuracy_score(y_test, rf_y_pred))
    # Create a SHAP TreeExplainer
    explainer = shap.TreeExplainer(rf_classifier)
    # Generate SHAP values
    shap_values = explainer.shap_values(X_test)
    # Plot the SHAP summary plot
    shap.summary_plot(shap_values, X_test)
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_tree(rf_classifier.estimators_[0], feature_names=list(df.columns)[
                                                          :-1], class_names=list(df.columns)[-1], filled=True, ax=ax)

    output_file = 'decision_tree.html'
    mpld3.save_html(fig, output_file)
    print(f"Interactive decision tree saved as: {output_file}")

def ensemble(X_train, y_train, X_test,y_test):
    """
    Perform ensemble classification using a Voting Classifier.

    Parameters:
        X_train (numpy.ndarray or pandas.DataFrame): The training data features.
        y_train (numpy.ndarray or pandas.Series): The training data labels.
        X_test (numpy.ndarray or pandas.DataFrame): The test data features.
        y_test (numpy.ndarray or pandas.Series): The test data labels.

    Returns:
        None

    Example:
        ensemble(X_train, y_train, X_test, y_test)
    """
    # We will perform a standard scalar to our training and testing data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize individual classifiers
    classifier_1 = RandomForestClassifier(random_state=42)
    classifier_2 = LogisticRegression(random_state=42)
    classifier_3 = SVC(random_state=42, probability=True)
    classifier_4 = MLPClassifier(solver='adam', alpha=1e-4,
                                 hidden_layer_sizes=(5, 2), random_state=1, max_iter=500)

    # Create the Voting Classifier
    voting_classifier = VotingClassifier(
        estimators=[('rf', classifier_1), ('lr', classifier_2),
                    ('svm', classifier_3), ("MLP", classifier_4)],
        voting='soft'
    )

    # Train the Voting Classifier
    voting_classifier.fit(X_train_scaled, y_train)

    y_pred = voting_classifier.predict(X_test_scaled)

    # Evaluate the Voting Classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Voting Accuracy: {accuracy}")


def train_all_models(X_train, X_test, y_train,y_test):
    """
    Train and evaluate multiple classification models.

    Parameters:
        X_train (numpy.ndarray or pandas.DataFrame): The training data features.
        X_test (numpy.ndarray or pandas.DataFrame): The test data features.
        y_train (numpy.ndarray or pandas.Series): The training data labels.
        y_test (numpy.ndarray or pandas.Series): The test data labels.

    Returns:
        None

    Example:
        train_all_models(X_train, X_test, y_train, y_test)
    """
    NB(X_train, y_train, X_test, y_test)
    ada(X_train, y_train, X_test, y_test)
    RF(X_train, y_train, X_test, y_test)
    ensemble(X_train, y_train, X_test, y_test)
