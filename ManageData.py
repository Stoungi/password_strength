import csv
import math
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


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
    overall_time = 0  # Class-level variable to keep track of overall time

    def __init__(self, fileName="no proper file given"):
        start_time = time.time()  # Start timing
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

        # Train Decision Tree model
        self.tree_model = DecisionTreeClassifier(random_state=42)
        self.tree_model.fit(self.X_train, self.y_train)

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        PassTrain.overall_time += elapsed_time  # Add to overall time



    def bestGuess(self, rating_confidences):
        return max(rating_confidences, key=rating_confidences.get)

    def score_password(self, password):
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

    def suggestImprovements(self, password):
        features = self._expand(password)
        strength = self.score_password(password)

        suggestions = []

        if strength == 0:
            if features[self.total_len] < 8:
                suggestions.append("Increase the length of your password to at least 8 characters.")
            if features[self.upper_case_count] == 0:
                suggestions.append("Add some uppercase letters.")
            if features[self.lower_case_count] == 0:
                suggestions.append("Add some lowercase letters.")
            if features[self.special_count] == 0:
                suggestions.append("Include special characters like @, #, $, etc.")
            if features[self.numbers] == 0:
                suggestions.append("Add some numbers.")
            if features[self.total_len] > 0 and features['unique_count'] > 0:
                ratio_len_unique = features['total_len'] / features['unique_count']
                if ratio_len_unique > 2.0:
                    suggestions.append("Increase character diversity: avoid repeating characters.")

        elif strength == 1:
            if features[self.total_len] < 14:
                suggestions.append("Consider increasing the length of your password further.")
            if features[self.upper_case_count] < 2:
                suggestions.append("Add a few more uppercase letters.")
            if features[self.lower_case_count] < 2:
                suggestions.append("Add a few more lowercase letters.")
            if features[self.special_count] < 2:
                suggestions.append("Include a couple more special characters. (@, #, $, etc)")
            if features[self.numbers] < 2:
                suggestions.append("Add a few more numbers.")
            if features[self.total_len] > 0 and features[self.unique_count] > 0:
                ratio_len_unique = features['total_len'] / features['unique_count']
                if ratio_len_unique > 2.0:
                    suggestions.append("Increase character diversity: avoid repeating characters.")

        else:
            suggestions.append(
                "Your password is strong. No improvements needed, but you can always add more complexity for even better security.")

        return suggestions




