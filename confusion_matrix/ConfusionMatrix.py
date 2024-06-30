import csv
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from time import time
#the link below is the unprocessed data set used
#https://www.kaggle.com/datasets/litvinenko630/password-correct/data

class PassData:
    def _expand(self, line):
        total_len = len(line)
        upper_case_count = sum(1 for char in line if char.isupper())
        lower_case_count = sum(1 for char in line if char.islower())
        special_count = sum(1 for char in line if not char.isalnum())
        numbers = sum(1 for char in line if char.isdigit())
        unique_count = len(set(line))

        return {
            'total_len': total_len,
            'upper_case_count': upper_case_count,
            'lower_case_count': lower_case_count,
            'special_count': special_count,
            'numbers': numbers,
            'unique_count': unique_count
        }

    def __init__(self, fileName="no proper file given"):
        self.total_len = "total_len"
        self.upper_case_count = "upper_case_count"
        self.lower_case_count = "lower_case_count"
        self.numbers = "numbers"
        self.special_count = "special_count"
        self.unique_count = "unique_count"
        self.strength = "strength"

        try:
            self._file = open(fileName, mode="r")
        except FileNotFoundError:
            while not fileName.endswith(".csv"):
                fileName = input("There was no .csv file given/could not find it, give a file path that exists\n~>>")
                try:
                    self._file = open(fileName, mode="r")
                except FileNotFoundError:
                    fileName = "no proper file given"

        self._content = csv.DictReader(self._file)
        self.contentS0 = []
        self.contentS1 = []
        self.contentS2 = []

        for row in self._content:
            try:
                row[self.total_len] = int(row[self.total_len])
                row[self.upper_case_count] = int(row[self.upper_case_count])
                row[self.lower_case_count] = int(row[self.lower_case_count])
                row[self.numbers] = int(row[self.numbers])
                row[self.special_count] = int(row[self.special_count])
                row[self.unique_count] = int(row[self.unique_count])
                row[self.strength] = int(row[self.strength])

                if row[self.strength] == 0:
                    self.contentS0.append(row)
                elif row[self.strength] == 1:
                    self.contentS1.append(row)
                elif row[self.strength] == 2:
                    self.contentS2.append(row)
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

        self.content = tuple(self.contentS0 + self.contentS1 + self.contentS2)
        self.contentS0 = tuple(self.contentS0)
        self.contentS1 = tuple(self.contentS1)
        self.contentS2 = tuple(self.contentS2)

    def getI(self, index=-1):
        if index == -1:
            return self.content
        else:
            return self.content[index]

    def getIByStrength(self, rating=-1):
        if rating == -1:
            return self.getI()
        if rating == 0:
            return self.contentS0
        if rating == 1:
            return self.contentS1
        if rating == 2:
            return self.contentS2

    def countByStrength(self, rating=-1):
        if rating == -1:
            return len(self.content)
        if rating == "avg":
            count = 0
            for row in self.content:
                count += row[self.strength]
            return math.ceil(count / len(self.content))
        if rating == 0:
            return len(self.contentS0)
        if rating == 1:
            return len(self.contentS1)
        if rating == 2:
            return len(self.contentS2)

    def getIByLen(self, leng=-1):
        if leng == -1:
            return self.content
        filt = []
        for row in self.content:
            if row[self.total_len] == leng:
                filt.append(row)
        return filt

    def countByLen(self, leng=-1):
        if leng == -1:
            return "no length given"

        elif leng == "avg":
            count = 0
            for row in self.content:
                count += row[self.total_len]
            return math.ceil(count / len(self.content))

        elif leng == "max":
            max_len = max(row[self.total_len] for row in self.content)
            return max_len

        elif leng == "min":
            min_len = min(row[self.total_len] for row in self.content)
            return min_len

        else:
            count = 0
            for row in self.content:
                if row[self.total_len] == leng:
                    count += 1
            return count

    def compareAndRate(self, password):
        pass

    def suggestImprovements(self, password):
        pass

    def bestGuess(self, rating_confidences):
        return max(rating_confidences, key=rating_confidences.get)


class PassTrain(PassData):
    def __init__(self, fileName="no proper file given"):
        super().__init__(fileName)
        self.X = []
        self.y = []

        for row in self.content:
            # Use only total_len and unique_count for features
            features = [
                row[self.total_len],
                row[self.unique_count],
                row[self.unique_count] / row[self.total_len]
            ]
            self.X.append(features)
            self.y.append(row[self.strength])

        # Create and fit StandardScaler
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

        # Train Logistic Regression model
        start_time = time()
        self.logreg_model = LogisticRegression(max_iter=1000, random_state=42)
        self.logreg_model.fit(self.X_train, self.y_train)
        self.logreg_time = time() - start_time
        self.logreg_accuracy = accuracy_score(self.y_test, self.logreg_model.predict(self.X_test))

        # Train Decision Tree model
        start_time = time()
        self.tree_model = DecisionTreeClassifier(random_state=42)
        self.tree_model.fit(self.X_train, self.y_train)
        self.tree_time = time() - start_time
        self.tree_accuracy = accuracy_score(self.y_test, self.tree_model.predict(self.X_test))

        # Train Random Forest model
        start_time = time()
        self.forest_model = RandomForestClassifier(random_state=42)
        self.forest_model.fit(self.X_train, self.y_train)
        self.forest_time = time() - start_time
        self.forest_accuracy = accuracy_score(self.y_test, self.forest_model.predict(self.X_test))

    def bestGuess(self, rating_confidences):
        return max(rating_confidences, key=rating_confidences.get)

    def score_password(self, password):
        # Get features for the password
        X_input = self.compareAndRate(password)

        # Transform X_input using the same StandardScaler instance
        X_input_scaled = self.scaler.transform([X_input])

        # Predict probabilities for each class using Logistic Regression model
        confidence_scores = {i: 0 for i in range(3)}

        # Get predictions from all models
        models = [self.logreg_model, self.tree_model, self.forest_model]
        for model in models:
            probabilities = model.predict_proba(X_input_scaled)[0]
            for strength_class, probability in enumerate(probabilities):
                confidence_scores[strength_class] += probability / len(models)

        return self.bestGuess(confidence_scores)

    def compareAndRate(self, password):
        features = self._expand(password)
        # Use only total_len and unique_count for features
        X_input = [
            features[self.total_len],
            features[self.unique_count],
            features[self.unique_count] / features[self.total_len]
        ]

        return X_input

    def display_metrics(self):
        print(
            f"Logistic Regression - Accuracy: {self.logreg_accuracy:.4f}, Training Time: {self.logreg_time:.4f} seconds")
        print(f"Decision Tree - Accuracy: {self.tree_accuracy:.4f}, Training Time: {self.tree_time:.4f} seconds")
        print(f"Random Forest - Accuracy: {self.forest_accuracy:.4f}, Training Time: {self.forest_time:.4f} seconds")

    def display_confusion_matrix(self):
        # Make predictions using the test set
        y_pred_logreg = self.logreg_model.predict(self.X_test)
        y_pred_tree = self.tree_model.predict(self.X_test)
        y_pred_forest = self.forest_model.predict(self.X_test)

        # Calculate confusion matrices
        cm_logreg = confusion_matrix(self.y_test, y_pred_logreg)
        cm_tree = confusion_matrix(self.y_test, y_pred_tree)
        cm_forest = confusion_matrix(self.y_test, y_pred_forest)

        # Plot confusion matrices
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        disp_logreg = ConfusionMatrixDisplay(confusion_matrix=cm_logreg, display_labels=[0, 1, 2])
        disp_logreg.plot(cmap=plt.cm.Blues, ax=plt.gca())
        plt.title("Logistic Regression Confusion Matrix")
        plt.savefig('logreg_confusion_matrix.png')  # Save plot as PNG

        plt.subplot(1, 3, 2)
        disp_tree = ConfusionMatrixDisplay(confusion_matrix=cm_tree, display_labels=[0, 1, 2])
        disp_tree.plot(cmap=plt.cm.Blues, ax=plt.gca())
        plt.title("Decision Tree Confusion Matrix")
        plt.savefig('tree_confusion_matrix.png')  # Save plot as PNG

        plt.subplot(1, 3, 3)
        disp_forest = ConfusionMatrixDisplay(confusion_matrix=cm_forest, display_labels=[0, 1, 2])
        disp_forest.plot(cmap=plt.cm.Blues, ax=plt.gca())
        plt.title("Random Forest Confusion Matrix")
        plt.savefig('forest_confusion_matrix.png')  # Save plot as PNG

        plt.tight_layout()
        # plt.show()  # Comment out plt.show() to prevent interactive display


# Create an instance of PassTrain with your dataset
pass_train = PassTrain("processed_data.csv")

# Display the confusion matrices (this will save them as PNG files)
pass_train.display_confusion_matrix()

# Display the accuracy and training time metrics
pass_train.display_metrics()

# it would seem that decision tree was faster after all

