import numpy as np
import pandas as pd
import pydantic
import typing
import databayes.modelling.DiscreteDistribution as dd
import databayes.modelling.DiscreteVariable as dv
import databayes.modelling.core as dcm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pkg_resources
installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb


class MLPClassifierFitParameters(dcm.FitParametersBase):
    pass


class MLPClassifierHyperParameters(dcm.HyperParametersBase):
    hidden_layer_sizes: tuple = pydantic.Field(
        (100,), description="The ith element represents the number of neurons in the ith hidden layer.")
    activation: str = pydantic.Field(
        'relu', description="Activation function for the hidden layer.")
    solver: str = pydantic.Field(
        'adam', description="The solver for weight optimization.")
    alpha: float = pydantic.Field(
        0.0001, description="L2 penalty (regularization term) parameter.")
    batch_size: typing.Union[int, str] = pydantic.Field(
        'auto', description="Size of minibatches for stochastic optimizers.")
    learning_rate: str = pydantic.Field(
        'constant', description="Learning rate schedule for weight updates")
    learning_rate_init: float = pydantic.Field(
        0.001, description="The initial learning rate used. It controls the step-size in updating the weights")
    power_t: float = pydantic.Field(
        0.5, description="The exponent for inverse scaling learning rate.")
    max_iter: int = pydantic.Field(
        200, description="Maximum number of iterations.")
    shuffle: bool = pydantic.Field(
        True, description="Whether to shuffle samples in each iteration.")
    random_state: typing.Any = pydantic.Field(
        None, description="Determines random number generation for weights and bias initialization")
    tol: float = pydantic.Field(
        1e-4, description="Tolerance for the optimization")
    verbose: bool = pydantic.Field(
        False, description="Whether to print progress messages to stdout.")
    warm_start: bool = pydantic.Field(
        False, description="When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.")
    momentum: float = pydantic.Field(
        0.9, description="Momentum for gradient descent update. Should be between 0 and 1.")
    nesterovs_momentum: bool = pydantic.Field(
        True, description="Whether to use Nesterov’s momentum")
    early_stopping: bool = pydantic.Field(
        False, description="Whether to use early stopping to terminate training when validation score is not improving")
    validation_fraction: float = pydantic.Field(
        0.1, description="The proportion of training data to set aside as validation set for early stopping")
    beta_1: float = pydantic.Field(
        0.9, description="Exponential decay rate for estimates of first moment vector in adam")
    beta_2: float = pydantic.Field(
        0.999, description="Exponential decay rate for estimates of second moment vector in adam")
    epsilon: float = pydantic.Field(
        1e-8, description="Value for numerical stability in adam.")
    n_iter_no_change: int = pydantic.Field(
        10, description="Maximum number of epochs to not meet tol improvement.")
    max_fun: int = pydantic.Field(
        15000, description="Only used when solver=’lbfgs’. Maximum number of loss function calls.")

    @pydantic.validator('activation')
    def check_activation(cls, val):
        if not (val in ['identity', 'logistic', 'tanh', 'relu']):
            raise ValueError(f"{val} isn't an activation function")
        return val

    @pydantic.validator('solver')
    def check_solver(cls, val):
        if not (val in ['lbfgs', 'sgd', 'adam']):
            raise ValueError(f"{val} isn't a solver for weight optimization")
        return val

    @pydantic.validator('learning_rate')
    def check_learning_rate(cls, val):
        if not (val in ['constant', 'invscaling', 'adaptive']):
            raise ValueError(f"{val} isn't a solver for weight optimization")
        return val


class RandomForestFitParameters(dcm.FitParametersBase):
    pass


class RandomForestHyperParameters(dcm.HyperParametersBase):
    n_estimators: int = pydantic.Field(
        100, description="The number of trees in the forest")
    criterion: str = pydantic.Field(
        "gini", description="The function to measure the quality of a split")  # {“gini”, “entropy”}
    max_depth: int = pydantic.Field(
        None, description="The maximum depth of the tree")
    min_samples_split: typing.Union[int, float] = pydantic.Field(
        2, description="The minimum number of samples required to split an internal node")
    min_samples_leaf: typing.Union[int, float] = pydantic.Field(
        1, description="The minimum number of samples required to be at a leaf node")
    min_weight_fraction_leaf: float = pydantic.Field(
        0.0, description="The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node")
    max_features: typing.Union[str, int, float] = pydantic.Field(
        "auto", description="The number of features to consider when looking for the best split")
    max_leaf_nodes: int = pydantic.Field(
        None, description="Grow trees with max_leaf_nodes in best-first fashion")
    min_impurity_decrease: float = pydantic.Field(
        0.0, description="A node will be split if this split induces a decrease of the impurity greater than or equal to this value")
    min_impurity_split: float = pydantic.Field(
        None, description="Threshold for early stopping in tree growth")
    bootstrap: bool = pydantic.Field(
        True, description="Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.")
    oob_score: bool = pydantic.Field(
        False, description="Whether to use out-of-bag samples to estimate the generalization accuracy")
    n_jobs: int = pydantic.Field(
        None, description="The number of jobs to run in parallel")
    random_state: int = pydantic.Field(None, description="")
    verbose: int = pydantic.Field(
        0, description="Controls the verbosity when fitting and predicting")
    warm_start: bool = pydantic.Field(
        False, description="When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest")
    class_weight: typing.Union[typing.List[dict],
                               dict, str] = pydantic.Field(None, description="")
    ccp_alpha: float = pydantic.Field(
        0.0, description="Complexity parameter used for Minimal Cost-Complexity Pruning")
    max_samples: typing.Union[int, float] = pydantic.Field(
        None, description="If bootstrap is True, the number of samples to draw from X to train each base estimator")


class MLSklearnClassifierModel(dcm.MLModel):

    column_mapping: typing.Dict[str, typing.Dict[typing.Union[str, int],
                                                 typing.Union[str, int]]] = \
        pydantic.Field(dict(),
                       description="Map the labels of the classifiers with int")

    def json(self, exclude=None, **kwargs):
        return super().json(exclude={"model"}, **kwargs)

    def dict(self, exclude=None, **kwargs):
        return super().dict(exclude={"model"}, **kwargs)

    def prepare_data(self, data):

        cols = self.var_features + self.var_targets
        for column in cols:

            if not(data[column].dtypes in ['float', 'int']):

                # key are labels, key are ind
                self.column_mapping[column] = {'label': dict(), 'ind': dict()}

                if not(isinstance(data[column].dtypes, pd.CategoricalDtype)):
                    data.loc[:, column] = data[column].astype(
                        'str').astype('category')

                category = data[column].cat.categories
                data.loc[:, column] = data[column].cat.codes

                for i, categ in enumerate(category):

                    self.column_mapping[column]['ind'][i] = categ
                    self.column_mapping[column]['label'][categ] = i

        X = data[self.var_features].to_numpy()
        y = data[self.var_targets].to_numpy()

        return X, y

    def transform_mapping(self, data, column_name=None, from_int_to_str=False):

        if not from_int_to_str:
            X = pd.DataFrame(np.zeros((len(data.index), len(
                data.columns))), index=data.index, columns=data.columns)

            for lbl, data_cur in data.iteritems():

                if data[lbl].dtypes in ['float', 'int']:
                    X[lbl] = data_cur
                else:
                    for data_id in data_cur.index:
                        if data_cur[data_id] in self.column_mapping[lbl]['label']:
                            X.loc[data_id, lbl] = self.column_mapping[lbl]['label'][data_cur[data_id]]
                        else:
                            # Si on n'a pas rencontré une valeur dans le train set, on set cette valeur à -1
                            X.loc[data_id, lbl] = -1

            X = X.to_numpy()

        else:

            n = len(data)
            X = [0 for i in range(n)]

            for l in range(n):
                X[l] = self.column_mapping[column_name]['ind'][data[l]]

        return X


class MLSklearnClassifierModelMultiLabel(MLSklearnClassifierModel):
    """Type of targets : multiclass-multioutput  cf sklearn doc."""

    def fit(self, data, logger=None, **kwds):

        X, y = self.prepare_data(data)

        if len(y[0]) == 1:
            y = y.ravel()
        self.model.fit(X, y,
                       **self.fit_parameters.dict(), **kwds)

    def predict(self, data, logger=None, progress_mode=False, **kwds):

        X = self.transform_mapping(data[self.var_features])

        y_pred_np = self.model.predict_proba(X,
                                             **self.predict_parameters.dict(), **kwds)

        pred_res = {tv:
                    {"scores": dd.DiscreteDistribution(index=data.index,
                                                       domain=list(self.column_mapping[tv]['label'].keys()))
                     }
                    for tv in self.var_targets}

        for i, tv in enumerate(self.var_targets):

            y_pred = y_pred_np if len(self.var_targets) == 1 else y_pred_np[i]
            domain_pred = self.model.classes_ if len(
                self.var_targets) == 1 else self.model.classes_[i]

            var_domain = self.transform_mapping(
                domain_pred, column_name=tv, from_int_to_str=True)

            pred_res[tv]["scores"].loc[:, var_domain] = dd.DiscreteDistribution(y_pred,
                                                                                index=data.index,
                                                                                domain=var_domain
                                                                                )
        return pred_res


class MLSklearnClassifierModelBinaryLabel(MLSklearnClassifierModel):
    """Type of targets : multilabel-indicator  cf sklearn doc"""

    model: typing.Dict[str, typing.Any] = pydantic.Field(
        None, description="Model storage structure")

    def fit(self, data, logger=None, **kwds):

        X, y = self.prepare_data(data)

        for i, tv in enumerate(self.var_targets):
            self.model[tv].fit(X, y[:, i].ravel(),
                               **self.fit_parameters.dict(), **kwds)

    def predict(self, data, logger=None, progress_mode=False, **kwds):

        X = self.transform_mapping(data[self.var_features])

        y_pred_dict = {tv: self.model[tv].predict_proba(X,
                                                        **self.predict_parameters.dict(), **kwds) for tv in self.var_targets}

        pred_res = dict()
        for tv in self.var_targets:

            # NOTE: Disjonction de cas selon le nombre de classe.
            # Si on n'a rencontré qu'un seul label dans le jeu de données, predict_proba() renvoit
            # un y_pred qui contient deux colonnes (alors qu'il ne devrait renvoyer qu'une seule colonne).
            # On crée donc une colonne unique de 1 si le nombre de classes est égal à 1

            if len(self.model[tv].classes_) > 1:
                pred_res[tv] = \
                    {"scores": dd.DiscreteDistribution(
                        y_pred_dict[tv],
                        index=data.index,
                        domain=self.transform_mapping(
                            self.model[tv].classes_, column_name=tv, from_int_to_str=True)
                    )
                }
            else:
                pred_res[tv] = \
                    {"scores": dd.DiscreteDistribution(
                        1,
                        index=data.index,
                        domain=self.transform_mapping(
                            self.model[tv].classes_, column_name=tv, from_int_to_str=True)
                    )
                }
        return pred_res


class RandomForestModel(MLSklearnClassifierModelMultiLabel):

    type: str = pydantic.Field(
        "RandomForestModel", description="Type of the model")

    model: typing.Any = pydantic.Field(
        RandomForestClassifier(), description="Bayesian network object")

    fit_parameters: RandomForestFitParameters = pydantic.Field(RandomForestFitParameters(),
                                                               description="")

    hyper_parameters: RandomForestHyperParameters = pydantic.Field(
        RandomForestHyperParameters(), description="")

    @ pydantic.validator('type')
    def check_type(cls, val):
        if val != "RandomForestModel":
            raise ValueError("Not RandomForestModel object")
        return val

    def __init__(self, **data: typing.Any):
        super().__init__(**data)
        self.model = RandomForestClassifier(**self.hyper_parameters.dict())


class MLPClassifierModel(MLSklearnClassifierModelBinaryLabel):

    type: str = pydantic.Field(
        "MLPClassifierModel", description="Type of the model")

    model: typing.Dict[str, typing.Any] = pydantic.Field(
        dict(), description="Bayesian network object")

    fit_parameters: MLPClassifierFitParameters = pydantic.Field(MLPClassifierFitParameters(),
                                                                description="")

    hyper_parameters: MLPClassifierHyperParameters = pydantic.Field(
        MLPClassifierHyperParameters(), description="")

    @ pydantic.validator('type')
    def check_type(cls, val):
        if val != "MLPClassifierModel":
            raise ValueError("Not MLPClassifierModel object")
        return val

    def __init__(self, **data: typing.Any):
        super().__init__(**data)
        self.model = {tv: MLPClassifier(
            **self.hyper_parameters.dict()) for tv in self.var_targets}
