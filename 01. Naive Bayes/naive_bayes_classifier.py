import numpy as np
import pandas as pd


class NaiveBayesClassifier:
    """
    Simple Naïve Bayes classifier (from scratch).
    Assumes the input is a pandas DataFrame where the last column
    contains the class labels (target variable).
    """

    def __init__(self):
        # Will store the unique class labels (e.g., ["British", "Scottish"])
        self.classes = None

        # Column names of the training dataframe
        self.columns = None

        # Name of the target column (last column of the dataframe)
        self.target_col = None

        # Conditional probabilities P(feature_value | class)
        # Stored as a nested list of dictionaries
        self.conditional_probs = None

        # Prior probabilities P(class)
        self.priors = None

        # Number of samples per class
        self.counts_per_class = None

    def fit(self, df):
        """
        Train the classifier from a dataframe (last column is the label).
        Vectorized version: uses groupby/value_counts and avoids per-value loops.
        """
        # Columns and target
        self.columns = df.columns
        self.target_col = self.columns[-1]

        # Class labels in a fixed order
        self.classes = df[self.target_col].unique()
        n_classes = len(self.classes)

        # Priors P(class)
        class_counts = (
            df[self.target_col]
            .value_counts()
            .reindex(self.classes, fill_value=0)
            .to_numpy()
            .astype(float)
        )
        n_samples = float(len(df))
        self.counts_per_class = class_counts
        self.priors = class_counts / n_samples

        # Prepare container for conditional probabilities
        n_features = len(self.columns) - 1
        self.conditional_probs = [[None] * n_features for _ in range(n_classes)]

        # Precompute possible values per feature once
        feature_values_per_col = {col: df[col].unique() for col in self.columns[:-1]}

        # For each feature, compute counts per class and value in one shot
        for j, col in enumerate(self.columns[:-1]):
            values = feature_values_per_col[col]

            # counts[class, value]
            ct = (
                df.groupby(self.target_col)[col]
                .value_counts()
                .unstack(fill_value=0)
            )

            # ensure all classes and all feature values are present
            ct = ct.reindex(index=self.classes, columns=values, fill_value=0)

            # Laplace smoothing and normalize per class
            for i, cls in enumerate(self.classes):
                denom = self.counts_per_class[i] + len(values)
                probs = (ct.loc[cls] + 1) / denom  # Series indexed by feature values
                self.conditional_probs[i][j] = probs.to_dict()


    def predict(self, attributes):
        """
        Classify a new instance.

        Parameters
        ----------
        attributes : list
            List of feature values (same order as DataFrame columns, excluding last).

        Returns
        -------
        predicted_class : str
            The most probable class label.
        evidence_df : pandas.DataFrame
            Conditional probabilities used in the prediction.
        posteriors : numpy.ndarray
            Posterior probabilities for each class.
        """
        # Initialize posterior probabilities with 1
        posteriors = np.ones(len(self.classes))

        # Evidence dictionary: stores the conditional probability
        # of each feature given each class
        evidence = {col: np.zeros(len(self.classes)) for col in self.columns[:-1]}

        # Compute posterior for each class
        for i, cls in enumerate(self.classes):
            # For each feature in the new instance
            for j, col in enumerate(self.columns[:-1]):
                val = attributes[j]

                # Retrieve P(feature_value | class)
                probs = self.conditional_probs[i][j]
                if val in probs:
                    # Multiply into posterior
                    posteriors[i] *= probs[val]

                    # Save for evidence table
                    evidence[col][i] = probs[val]

            # Multiply by the prior probability P(class)
            posteriors[i] *= self.priors[i]

        # The predicted class is the one with the highest posterior
        predicted_idx = np.argmax(posteriors)
        predicted_class = self.classes[predicted_idx]

        # Convert evidence dictionary to DataFrame for readability
        evidence_df = pd.DataFrame.from_dict(evidence)

        return predicted_class, evidence_df, posteriors
