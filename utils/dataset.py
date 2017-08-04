import numpy as np
from csv import DictReader


class DataSet(object):

    def __init__(self, csv_path, vector_cols, label_col):
        self.csv_path = csv_path
        self.vector_cols = vector_cols
        self.label_col = label_col

        self.input_set = []
        self.input_labels = []
        self._load()

        self.labels = list(set(self.input_labels))
        self.vector_len = len(self.labels)
        self.output_set = [
            self.vector(label)
            for label in self.input_labels
        ]

    @property
    def dim_in(self):
        return len(self.vector_cols)

    @property
    def dim_out(self):
        return len(self.labels)

    def display(self):
        for vector, output in zip(self.input_set, self.output_set):
            print(vector, "|", output)

    def _load(self):
        """Load datas in file"""
        with open(self.csv_path) as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                self.input_set.append(np.array([
                    float(row[col])
                    for col in self.vector_cols
                ]))
                self.input_labels.append(row[self.label_col])

    def vector(self, label):
        try:
            index = self.labels.index(label)
        except ValueError:
            raise Exception("Unknown Label '%s'" % label)
        vector = np.zeros(self.vector_len)
        vector[index] = 1
        return vector

    def labelize(self, vector_or_list):
        if isinstance(vector_or_list, np.ndarray):
            vector = vector_or_list
            maximum = max(vector)
            for i, vector_i in enumerate(vector):
                if vector_i == maximum:
                    return self.labels[i]
        else:
            vector_list = vector_or_list
            return [self.labelize(vector) for vector in vector_list]

    def rights_ratio(self, predictions, base=None):
        if base is None:
            base = self.output_set
        return np.mean([
            self.labelize(prediction) == self.labelize(ref)
            for prediction, ref in zip(predictions, base)
        ])
