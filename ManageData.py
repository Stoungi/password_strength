import csv
import math
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Isearch(ABC):
    @abstractmethod
    def getBy(self, criteria: Dict[str, Any], bool) -> List[Dict[str, Any]]:
        pass

class Icount(ABC):
    @abstractmethod
    def countBy(self, criteria: Dict[str, Any], bool) -> int:
        pass


class PassData(Isearch, Icount, ABC):
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

        self._total_len = "total_len"
        self._upper_case_count = "upper_case_count"
        self._lower_case_count = "lower_case_count"
        self._numbers = "numbers"
        self._special_count = "special_count"
        self._unique_count = "unique_count"
        self._strength = "strength"

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
                row[self._total_len] = int(row[self._total_len])
                row[self._upper_case_count] = int(row[self._upper_case_count])
                row[self._lower_case_count] = int(row[self._lower_case_count])
                row[self._numbers] = int(row[self._numbers])
                row[self._special_count] = int(row[self._special_count])
                row[self._unique_count] = int(row[self._unique_count])
                row[self._strength] = int(row[self._strength])

                if row[self._strength] == 0:
                    self.contentS0.append(row)
                elif row[self._strength] == 1:
                    self.contentS1.append(row)
                elif row[self._strength] == 2:
                    self.contentS2.append(row)
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

        self.content = tuple(self.contentS0 + self.contentS1 + self.contentS2)
        self.contentS0 = tuple(self.contentS0)
        self.contentS1 = tuple(self.contentS1)
        self.contentS2 = tuple(self.contentS2)

    def getBy(self, **criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        def matches(row: Dict[str, Any]) -> bool:
            for attr, value in criteria.items():
                if row.get(attr) != value:
                    return False
            return True

        return [row for row in self.content if matches(row)]

    def countBy(self, **criteria: Dict[str, Any]) -> int:
        return len(self.getBy(**criteria))



    @abstractmethod
    def compareAndRate(self, password):
        pass

    @abstractmethod
    def suggestImprovements(self, password):
        pass

    def bestGuess(self, rating_confidences):
        return max(rating_confidences, key=rating_confidences.get)


class PassTrain(PassData):
    overall_time = 0  # Class-level variable to keep track of overall time

    def __init__(self, fileName="no proper file given", input=-1):
        start_time = time.time()  # Start timing
        super().__init__(fileName)
        self.X = []
        self.y = []

        for row in self.content:
            # Use only total_len and unique_count for features
            self.improve = ""
            self.score = 0
            self.history = []
            features = [
                row[self._total_len],
                row[self._unique_count],
                (row[self._unique_count] / row[self._total_len]) * 2
            ]
            self.X.append(features)
            self.y.append(row[self._strength])

        # Create and fit StandardScaler
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

        # Train Decision Tree model
        self.tree_model = DecisionTreeClassifier(random_state=42)
        self.tree_model.fit(self.X_train, self.y_train)

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        PassTrain.overall_time += elapsed_time  # Add to overall time

        if input != -1:
            self.score_password(input, True)
            self.suggestImprovements(input)

      

    def bestGuess(self, rating_confidences):
        return max(rating_confidences, key=rating_confidences.get)

    def score_password(self, password, do_guess = False):
        start_time = time.time()  # Start timing


        # Get features for the password
        X_input = self.compareAndRate(password)

        # Transform X_input using the same StandardScaler instance
        X_input_scaled = self.scaler.transform([X_input])

        # Predict probabilities for each class using Decision Tree model
        confidence_scores = {}
        probabilities = self.tree_model.predict_proba(X_input_scaled)[0]
        for strength_class, probability in enumerate(probabilities):
            confidence_scores[strength_class] = probability

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        PassTrain.overall_time += elapsed_time  # Add to overall time
        self.score = self.bestGuess(confidence_scores)
        self.suggestImprovements(password)
        if do_guess:
            return self.score
        else:
            return confidence_scores

    def compareAndRate(self, password):
        features = self._expand(password)
        # Use only total_len and unique_count for features
        X_input = [
            features[self._total_len],
            features[self._unique_count],
            features[self._unique_count] / features[self._total_len]
        ]

        return X_input

    def suggestImprovements(self, password):
        features = self._expand(password)
        strength = self.score

        suggestions = []

        if strength == 0:
            if features[self._total_len] < 8:
                suggestions.append(" Increase the length of your password to at least 8 characters. ")
            if features[self._upper_case_count] == 0:
                suggestions.append(" Add some uppercase letters. ")
            if features[self._lower_case_count] == 0:
                suggestions.append(" Add some lowercase letters. ")
            if features[self._special_count] == 0:
                suggestions.append(" Include special characters like @, #, $, etc. ")
            if features[self._numbers] == 0:
                suggestions.append(" Add some numbers. ")
            if features[self._total_len] > 0 and features['unique_count'] > 0:
                ratio_len_unique = features['total_len'] / features['unique_count']
                if ratio_len_unique > 2.0:
                    suggestions.append(" Increase character diversity: avoid repeating characters. ")

        elif strength == 1:
            if features[self._total_len] < 14:
                suggestions.append(" Consider increasing the length of your password further. ")
            if features[self._upper_case_count] < 2:
                suggestions.append(" Add a few more uppercase letters. ")
            if features[self._lower_case_count] < 2:
                suggestions.append(" Add a few more lowercase letters. ")
            if features[self._special_count] < 2:
                suggestions.append(" Include a couple more special characters. (@, #, $, etc) ")
            if features[self._numbers] < 2:
                suggestions.append(" Add a few more numbers. ")
            if features[self._total_len] > 0 and features[self._unique_count] > 0:
                ratio_len_unique = features['total_len'] / features['unique_count']
                if ratio_len_unique > 2.0:
                    suggestions.append(" Increase character diversity: avoid repeating characters. ")

        else:
            suggestions.append(
                " Your password is strong. No improvements needed, but you can always add more complexity for even better security. ")



        self.improve = suggestions
        self.history.append([password, self.improve])
        return suggestions
