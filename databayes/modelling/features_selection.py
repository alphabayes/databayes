import numpy as np
import pandas as pd
import pydantic
import typing
import datautils_common.utils.ml_performance as mlperf
import datautils_common.modelling.core as dcm
from sklearn.neighbors import KDTree

import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


class FeaturesEvaluation(pydantic.BaseModel):
    model: dcm.MLModel = pydantic.Field(
        dcm.MLModel(), description="Need a model for the wrapper methods")
    name: str = pydantic.Field("", description="Feature selection method")
    features_set: list = pydantic.Field(...,
                                        description="Select features among an initial set")
    targets_set: list = pydantic.Field(...,
                                       description="Select targets among an initial set")

    def evaluate_scores(self, data, var_selection_option="union"):
        pass

    def select_features(self, method_info, var_selection_option="union", **kwargs):

        features_selected = dict()
        if var_selection_option == 'union':
            for var_target, features in method_info.items():
                for feature, score in features.items():
                    if not feature in features_selected:
                        features_selected[feature] = score
        elif var_selection_option == 'inter':
            for var_target, features in method_info.items():
                for feature, score in features.items():
                    if not feature in features_selected:
                        features_selected[feature] = score
            for var_target, features in method_info.items():
                for feature in self.features_selection_parameters.features_set:
                    if not feature in features.keys() and feature in features_selected:
                        features_selected.pop(feature)
        elif var_selection_option in self.features_selection_parameters.targets_set:
            features_selected = method_info[var_selection_option]

        return features_selected


class FeaturesEvaluationResults(pydantic.BaseModel):
    name: str = pydantic.Field("", description="")
    scores: typing.Dict[str, float] = pydantic.Field(dict(), description="")

    def select(self, threshold, sort='ascending'):
        if sort == 'ascending':
            selected_variables = [
                var for var, score in self.scores.items() if score >= threshold]
        elif sort == 'descending':
            selected_variables = [
                var for var, score in self.scores.items() if score <= threshold]
        return selected_variables


class FCBFResults(FeaturesEvaluationResults):
    var_scores: typing.Dict[str, typing.Dict[str, float]
                            ] = pydantic.Field(dict(), description="")


class FCBF(FeaturesEvaluation):

    """
    fcbf_thresh : float
        A value in [0,1) used as threshold for selecting 'relevant' features. 
        A negative value suggest the use of minimum SU[i,c] value as threshold.
    """

    # Negative value => minimum SU
    fcbf_thresh: float = pydantic.Field(-1, description="FCBF thresh")

    @staticmethod
    def entropy(vec, base=2):
        " Returns the empirical entropy H(X) in the input vector."
        _, vec = np.unique(vec, return_counts=True)
        prob_vec = np.array(vec/float(sum(vec)))
        if base == 2:
            logfn = np.log2
        elif base == 10:
            logfn = np.log10
        else:
            logfn = np.log
        return prob_vec.dot(-logfn(prob_vec))

    @staticmethod
    def conditional_entropy(x, y):
        "Returns H(X|Y)."
        uy, uyc = np.unique(y, return_counts=True)
        prob_uyc = uyc/float(sum(uyc))
        cond_entropy_x = np.array([FCBF.entropy(x[y == v]) for v in uy])
        return prob_uyc.dot(cond_entropy_x)

    @staticmethod
    def mutual_information(x, y):
        " Returns the information gain/mutual information [H(X)-H(X|Y)] between two random vars x & y."
        return FCBF.entropy(x) - FCBF.conditional_entropy(x, y)

    @staticmethod
    def symmetrical_uncertainty(x, y):
        " Returns 'symmetrical uncertainty' (SU) - a symmetric mutual information measure."
        return 2.0*FCBF.mutual_information(x, y)/(FCBF.entropy(x) + FCBF.entropy(y))

    @staticmethod
    def getFirstElement(d):
        """
        Returns tuple corresponding to first 'unconsidered' feature

        Parameters:
        ----------
        d : ndarray
                A 2-d array with SU, original feature index and flag as columns.

        Returns:
        -------
        a, b, c : tuple
                a - SU value, b - original feature index, c - index of next 'unconsidered' feature
        """

        t = np.where(d[:, 2] > 0)[0]
        if len(t):
            return d[t[0], 0], d[t[0], 1], t[0]
        return None, None, None

    @staticmethod
    def getNextElement(d, idx):
        """
        Returns tuple corresponding to the next 'unconsidered' feature.

        Parameters:
        -----------
        d : ndarray
                A 2-d array with SU, original feature index and flag as columns.
        idx : int
                Represents original index of a feature whose next element is required.

        Returns:
        --------
        a, b, c : tuple
                a - SU value, b - original feature index, c - index of next 'unconsidered' feature
        """
        t = np.where(d[:, 2] > 0)[0]
        t = t[t > idx]
        if len(t):
            return d[t[0], 0], d[t[0], 1], t[0]
        return None, None, None

    @staticmethod
    def removeElement(d, idx):
        """
        Returns data with requested feature removed.

        Parameters:
        -----------
        d : ndarray
                A 2-d array with SU, original feature index and flag as columns.
        idx : int
                Represents original index of a feature which needs to be removed.

        Returns:
        --------
        d : ndarray
                Same as input, except with specific feature removed.
        """
        d[idx, 2] = 0
        return d

    @staticmethod
    def c_correlation(X, y):
        """
        Returns SU values between each feature and class.

        Parameters:
        -----------
        X : 2-D ndarray
                Feature matrix.
        y : ndarray
                Class label vector

        Returns:
        --------
        su : ndarray
                Symmetric Uncertainty (SU) values for each feature.
        """
        su = np.zeros(X.shape[1])
        for i in np.arange(X.shape[1]):
            su[i] = FCBF.symmetrical_uncertainty(X[:, i], y)
        return su

    @staticmethod
    def fcbf(X, y, thresh):
        """
        Perform Fast Correlation-Based Filter solution (FCBF).

        Parameters:
        -----------
        X : 2-D ndarray
                Feature matrix
        y : ndarray
                Class label vector
        thresh : float
                A value in [0,1) used as threshold for selecting 'relevant' features. 
                A negative value suggest the use of minimum SU[i,c] value as threshold.

        Returns:
        --------
        sbest : 2-D ndarray
                An array containing SU[i,c] values and feature index i.
        """
        n = X.shape[1]
        slist = np.zeros((n, 3))
        slist[:, -1] = 1

        # identify relevant features
        slist[:, 0] = FCBF.c_correlation(X, y)  # compute 'C-correlation'
        idx = slist[:, 0].argsort()[::-1]
        slist = slist[idx, ]
        slist[:, 1] = idx
        if thresh < 0:
            thresh = np.median(slist[-1, 0])
            #print("Using minimum SU value as default threshold: {0}".format(thresh))
        elif thresh >= 1 or thresh > max(slist[:, 0]):
            print("No relevant features selected for given threshold.")
            print("Please lower the threshold and try again.")
            exit()

        slist = slist[slist[:, 0] > thresh, :]  # desc. ordered per SU[i,c]

        # identify redundant features among the relevant ones
        cache = {}
        m = len(slist)
        p_su, p, p_idx = FCBF.getFirstElement(slist)
        for i in range(m):
            p = int(p)
            q_su, q, q_idx = FCBF.getNextElement(slist, p_idx)
            if q:
                while q:
                    q = int(q)
                    if (p, q) in cache:
                        pq_su = cache[(p, q)]
                    else:
                        pq_su = FCBF.symmetrical_uncertainty(X[:, p], X[:, q])
                        cache[(p, q)] = pq_su

                    if pq_su >= q_su:
                        slist = FCBF.removeElement(slist, q_idx)
                    q_su, q, q_idx = FCBF.getNextElement(slist, q_idx)

            p_su, p, p_idx = FCBF.getNextElement(slist, p_idx)
            if not p_idx:
                break

        sbest = slist[slist[:, 2] > 0, :2]
        return sbest

    def evaluate_scores(self, data, var_selection_option='union'):
        """
        SU_dict is a dict {var_target: {feature: score}}

        Main function call to perform FCBF selection. Saves Symmetric Uncertainty (SU)
        values and 0-based indices of selected features to a CSV file at the same location
        as input file, with 'feature_' as prefix. e.g. 'feature_pima.csv' for 'pima.csv'.
        """
        data = data[self.features_set+self.targets_set]
        SU_dict = dict()
        for var_target in self.targets_set:

            X = data.drop(self.targets_set, axis=1).to_numpy()
            y = data[var_target].to_numpy()

            #print("Performing FCBF selection. Please wait ...")
            #print('X: {}, y: {}'.format(X.shape, len(y)))
            sbest = FCBF.fcbf(X, y, self.fcbf_thresh)
            #print(f"Target: {var_target}")
            #print("Selected feature indices:\n{0}".format(sbest))
            # print("Selected feature:\n{0}".format(
            #    data.columns[sbest[:, 1].astype(int)].tolist()))

            SU_dict[var_target] = dict()
            for i in range(len(sbest)):
                variable_name = data.columns[sbest[i, 1].astype(int)]
                SU_dict[var_target][variable_name] = sbest[i, 0]

        return FCBFResults(**{'name': 'FCBF', 'scores': self.select_features(SU_dict, var_selection_option), 'var_scores': SU_dict})


class ExceptOne(FeaturesEvaluation):

    # fit_parameters:

    def evaluate_scores(self, data):
        """
        Returns a dict {feature: score}
        The highest scores correspond to the features we should remove
        """
        data = data[self.features_set+self.targets_set]

        scores = dict()
        for feature in self.features_set:
            current_model = self.model.change_var_features(
                [feature], inplace=False)
            if len(self.targets_set) > 1:
                measures = {
                    'joined_accuracy': mlperf.JoinedAccuracyParameters()}
            else:
                measures = {'accuracy': mlperf.AccuracyParameters()}

            initialize_fit_parameters = {'percentage_training_data': 0.9,
                                         'training_sliding_window_size': 1,
                                         'testing_sliding_window_size': 1,
                                         'is_test_pct': True,
                                         'is_train_pct': True}

            initialize_mlperf = {'model': current_model,
                                 'measures': measures,
                                 'fit_parameters': mlperf.FitParameters(**initialize_fit_parameters)}

            ml_perf = mlperf.MLPerformance(**initialize_mlperf)
            compute_perf = ml_perf.run(data)

            if len(self.targets_set) > 1:
                scores[feature] = compute_perf['joined_accuracy'][0]['nb_pred_average']
            else:
                scores[feature] = compute_perf['accuracy'][0]['accuracy']
        return FeaturesEvaluationResults(**{'name': 'ExceptOne', 'scores': scores})


class ReliefF(FeaturesEvaluation):

    # PAS FINI DU TOUT
    """Feature selection using data-mined expert knowledge.
    Based on the ReliefF algorithm as introduced in:
    Kononenko, Igor et al. Overcoming the myopia of inductive learning
    algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55
    """

    feature_scores: typing.Any = pydantic.Field(
        None, description="Features scores")
    top_features: typing.Any = pydantic.Field(None, description="Top features")
    n_neighbors: int = pydantic.Field(100, description="Top features")
    n_features_to_keep: int = pydantic.Field(10, description="Top features")

    def fit(self, X, y):
        """Computes the feature importance scores from the training data.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        }
        Returns
        -------
        None
        """
        self.feature_scores = np.zeros(X.shape[1], dtype=np.int64)
        tree = KDTree(X)

        # Find nearest k neighbors of all points. The tree contains the query
        # points, so we discard the first match for all points (first column).
        indices = tree.query(X, k=self.n_neighbors+1,
                             return_distance=False)[:, 1:]

        for (source, nn) in enumerate(indices):
            # Create a binary array that is 1 when the sample  and neighbors
            # match and -1 everywhere else, for labels and features.
            labels_match = np.equal(y[source], y[nn]) * 2 - 1
            features_match = np.equal(X[source], X[nn]) * 2 - 1

            # The change in feature_scores is the dot product of these  arrays
            self.feature_scores += np.dot(features_match.T, labels_match)

        # Compute indices of top features, cast scores to floating point.
        self.top_features = np.argsort(self.feature_scores)[::-1]
        self.feature_scores = self.feature_scores.astype(np.float64)

    def transform(self, X):
        """Reduces the feature set down to the top `n_features_to_keep` features.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on
        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix
        """
        return X[:, self.top_features[:self.n_features_to_keep]]

    def fit_transform(self, X, y):
        """Computes the feature importance scores from the training data, then
        reduces the feature set down to the top `n_features_to_keep` features.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix
        """
        self.fit(X, y)
        return self.transform(X)

    def select(self, data, var_target):

        data = data[self.features_set+[var_target]]

        X = data.drop([var_target], axis=1).to_numpy()
        y = data[var_target].to_numpy()
        print("Performing ReliefF selection. Please wait ...")
        print('X: {}, y: {}'.format(X.shape, len(y)))
        self.fit(X, y)
        scores = pd.Series(self.feature_scores, index=X.columns)
        print("Done!")
        print("\n#Scores: {0}".format(scores))
        #print("Selected feature indices:\n{0}".format())
        #print("Selected feature:\n{0}".format())
