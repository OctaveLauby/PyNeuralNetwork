import numpy as np
from csv import DictReader


class BaseDataSet(object):

    def __init__(self, input_set, input_labels, dim_in, labels=None,
                 std_scale=False):
        self.input_set = input_set
        self.input_labels = input_labels

        self._dim_in = dim_in

        if labels:
            self.labels = labels
        else:
            self.labels = list(set(self.input_labels))
            self.labels.sort()
        self.vector_len = len(self.labels)
        self.output_set = [
            self.vectorize(label)
            for label in self.input_labels
        ]

        if std_scale:
            self.std_scale()

        assert len(self.input_labels) == len(self.output_set)
        for vector in self.input_set:
            assert len(vector) == self.dim_in

        for vector in self.output_set:
            assert len(vector) == self.dim_out

    @property
    def dim_in(self):
        return self._dim_in

    @property
    def dim_out(self):
        return len(self.labels)

    @property
    def size(self):
        return len(self.input_set)

    def display_data(self):
        for vector, output in zip(self.input_set, self.output_set):
            print(vector, "|", output)

    def display_struct(self):
        print("DataSet :")
        print("\tLabels     :", ", ".join([
            "%s (%s)" % (label, self.input_labels.count(label))
            for label in self.labels
        ]))
        print("\tSize       :", self.size)

    def labelize(self, vector_or_list):
        """Return label of vector or list of vectors."""
        if isinstance(vector_or_list, np.ndarray):
            vector = vector_or_list
            maximum = max(vector)
            for i, vector_i in enumerate(vector):
                if vector_i == maximum:
                    return self.labels[i]
        else:
            vector_list = vector_or_list
            return [self.labelize(vector) for vector in vector_list]

    def vectorize(self, label):
        """Return reference vector associated to label."""
        try:
            index = self.labels.index(label)
        except ValueError:
            raise Exception("Unknown Label '%s'" % label)
        vector = np.zeros(self.vector_len)
        vector[index] = 1
        return vector

    def rights_ratio(self, predictions, base=None):
        """Return % of rights."""
        if base is None:
            base = self.output_set
        return np.mean([
            self.labelize(prediction) == self.labelize(ref)
            for prediction, ref in zip(predictions, base)
        ])

    def split(self, ratio):
        """Split data set in 2, given a ratio.

        Respects proportion of each labels.
        """
        training_vectors = []
        training_labels = []
        generalisation_vectors = []
        generalisation_labels = []

        for label in self.labels:
            label_n = self.input_labels.count(label)
            middle = int(ratio * label_n)

            count = 0
            index = 0
            while count < middle:
                if self.input_labels[index] == label:
                    training_vectors.append(self.input_set[index])
                    training_labels.append(label)
                    count += 1
                index += 1

            while count < label_n:
                if self.input_labels[index] == label:
                    generalisation_vectors.append(self.input_set[index])
                    generalisation_labels.append(label)
                    count += 1
                index += 1

        return (
            BaseDataSet(
                training_vectors,
                training_labels,
                dim_in=self.dim_in,
                labels=self.labels,
                std_scale=False,
            ),
            BaseDataSet(
                generalisation_vectors,
                generalisation_labels,
                dim_in=self.dim_in,
                labels=self.labels,
                std_scale=False,
            ),
        )

    def stats(self):
        """Return stats per column."""
        vectors = np.array(self.input_set)
        return {
            'mean': np.array([
                np.mean(vectors[:, i]) for i in range(self.dim_in)
            ]),
            'std_dev': np.array([
                np.std(vectors[:, i]) for i in range(self.dim_in)
            ]),
        }

    def std_scale(self):
        stats = self.stats()
        self.input_set = (
            (self.input_set - stats['mean'])
            / stats['std_dev']
        )


class DataSet(BaseDataSet):

    def __init__(self, csv_path, label_col, vector_cols=None, std_scale=False):
        """Create dataset.

        Args:
            csv_path (str):     path to csv
            label_col (str):    column name where to read label
            vector_cols (str):  names of columns to build vectors
                > default is all columns except label col
            std_scale (bool):   use z-score scaling
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
        with open(csv_path) as csvfile:
            reader = DictReader(csvfile)
            assert self.label_col in reader.fieldnames
            for vector_col in self.vector_cols:
                assert vector_col in reader.fieldnames

        self.input_set = []
        self.input_labels = []
        self._load()

        super().__init__(
            self.input_set,
            self.input_labels,
            dim_in=len(self.vector_cols),
            std_scale=std_scale,
        )

    def display_struct(self):
        print("DataSet :", self.csv_path)
        print("\tInput vect :", self.vector_cols)
        print("\tLabels     :", ", ".join([
            "%s (%s)" % (label, self.input_labels.count(label))
            for label in self.labels
        ]))
        print("\tSize       :", self.size)

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
