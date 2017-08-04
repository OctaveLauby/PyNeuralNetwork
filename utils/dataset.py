import numpy as np
from csv import DictReader


class DataSet(object):

    def __init__(self, csv_path, label_col, vector_cols=None):
        """Create dataset.

        Args:
            csv_path (str):     path to csv
            label_col (str):    column name where to read label
            vector_cols (str):  names of columns to build vectors
                > default is all columns except label col
        """
        self.csv_path = csv_path
        self.label_col = label_col
        if vector_cols:
            self.vector_cols = vector_cols
        else:
            with open(csv_path) as csvfile:
                reader = DictReader(csvfile)
                self.vector_cols = [
                    field for field in reader.fieldnames if field != label_col
                ]
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

    def display_data(self):
        for vector, output in zip(self.input_set, self.output_set):
            print(vector, "|", output)

    def display_struct(self):
        print("DataSet :", self.csv_path)
        print("\tInput vect :", self.vector_cols)
        print("\tLabels     :", ", ".join([
            "%s (%s)" % (label, self.input_labels.count(label))
            for label in self.labels
        ]))
        print("\tSize       :", len(self.input_set))

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
