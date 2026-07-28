import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def prepare_data(file_path):
    data = pd.read_csv(file_path, index_col=0)

    data["Admitted"] = data["Chance of Admit"] >= 0.75
    data = data.drop(["Chance of Admit"], axis=1)

    features = data.drop(["Admitted"], axis=1)
    labels = data["Admitted"]
    return features, labels


def train_model(features, labels):
    model = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=10, min_samples_split=10
    )
    model.fit(features, labels)
    return model


if __name__ == "__main__":
    X, y = prepare_data("Admission_Predict.csv")

    dt_model = train_model(X, y)

    accuracy = dt_model.score(X, y)
    print(f"The model has been trained. Accuracy on the training data: {accuracy:.3%}")

    # Criteria: GRE, TOEFL, UnivRating, SOP, LOR, CGPA, Research
    sample_student = pd.DataFrame([[320, 110, 3, 4.0, 3.5, 8.9, 0]], columns=X.columns)
    prediction = dt_model.predict(sample_student)

    result = "Accepted" if prediction else "Not accepted"
    print(f"A prediction for a student: {result}")
