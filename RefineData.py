import pandas as pd




class PassRefineSet:


    def _expand(self, line):
        total_len = len(line)
        upper_case_count = sum(1 for char in line if char.isupper())
        lower_case_count = sum(1 for char in line if char.islower())
        special_count = sum(1 for char in line if not char.isalpha() and not char.isdigit())
        numbers = sum(1 for char in line if char.isdigit())
        unique_count = len(set(line))


        return total_len, upper_case_count, lower_case_count, special_count, numbers, unique_count

    def __init__(self):
        self._content = None
        self._file = None
        self.contentS0 = []
        self.contentS1 = []
        self.contentS2 = []

    def process_data(self, dataset):
        self._file = pd.read_csv(dataset)
        self._content = self._file.to_dict(orient='records')

        for row in self._content:
            try:
                if 0 < len(row["password"]):
                    if int(row['strength']) == 0:
                        self.contentS0.append((self._expand(row['password']), int(row['strength'])))
                    elif int(row['strength']) == 1:
                        self.contentS1.append((self._expand(row['password']), int(row['strength'])))
                    elif int(row['strength']) == 2:
                        self.contentS2.append((self._expand(row['password']), int(row['strength'])))
            except:
                continue

        all_content = self.contentS0 + self.contentS1 + self.contentS2
        expanded_data = [item[0] + (item[1],) for item in all_content]

        columns = [
            'total_len', 'upper_case_count', 'lower_case_count',
            'special_count', 'numbers', 'unique_count', 'strength'
        ]

        result_df = pd.DataFrame(expanded_data, columns=columns)
        return result_df

    def save_to_csv(self, output_file, dataset):
        result_df = self.process_data(dataset)
        if result_df is not None:
            result_df.to_csv(output_file, index=False)
            print(f"Processed data saved to {output_file}")