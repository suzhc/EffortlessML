import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    chi2,
    SequentialFeatureSelector,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    r2_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline



RANDOM_STATE = 42


class ExpML:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        model_name: str,
        feature_selector_name: str,
        n_folds: int = 5,
        keep_features: int = 20,
    ):
        self.X = X
        self.y = y
        self.keep_features = keep_features
        self.model = self.__get_model(model_name)
        self.feature_selector = self.__get_feature_selector(feature_selector_name)
        print("----------------")
        print(
            f"Label: {y.name}, Model: {model_name}"
            f", Selector: {feature_selector_name}, Folds: {n_folds}"
        )
        print(
            "Features count", len(X.columns)
        )

        self.run_exp(n_folds=n_folds)

    def __get_model(self, model_name: str):
        model_dict = {
            "svm": svm.SVC(kernel="rbf", probability=True),
            "mlp": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation="relu",
                solver="adam",
                alpha=0.3,
                random_state=RANDOM_STATE,
            ),
            "lr": LogisticRegression(random_state=RANDOM_STATE),
            "rf": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            "gb": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
            "dt": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "knn": KNeighborsClassifier(n_neighbors=5)
        }
        return model_dict[model_name]

    def __get_feature_selector(self, selector_name: str):
        kn = KNeighborsClassifier(n_neighbors=3)
        selector_dict = {
            "all": None,
            "fvalue": SelectKBest(f_classif, k=self.keep_features),
            "sequential": SequentialFeatureSelector(
                kn, n_features_to_select=self.keep_features
            ),
        }
        return selector_dict[selector_name]

    def run_exp(self, n_folds: int):
        X = self.X
        y = self.y

        feat_cols = X.columns
        folds = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE
        )

        auc_vals = []
        acc_vals = []
        prec_vals = []
        recall_vals = []
        f1_vals = []

        fpr_list = []
        tpr_list = []
        auc_list = []
        cm_list = []

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]

            pipeline = Pipeline(
                [
                    ("scalar", StandardScaler()),
                    ("feature_selector", self.feature_selector),
                    ("classifier", self.model),  # Model training step
                ]
            )
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_val)
            y_pred_prob = pipeline.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
            roc_auc = roc_auc_score(y_val, y_pred_prob)
            cm = confusion_matrix(y_val, y_pred)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(roc_auc)
            cm_list.append(cm)

            acc = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average="binary")
            recall = recall_score(y_val, y_pred, average="binary")
            f1 = f1_score(y_val, y_pred, average="binary")
            print(
                f"Fold: {n_fold}, acc is: {round(acc, 3)}, ",
                f"precision is {round(precision, 3)}, ",
                f"recall is {round(recall, 3)}, f1 is {round(f1, 3)}, ",
                f"ROC is: {round(roc_auc, 3)}",
            )

            auc_vals.append(roc_auc)
            acc_vals.append(acc)
            prec_vals.append(precision)
            recall_vals.append(recall)
            f1_vals.append(f1)

        self.fpr_list = fpr_list
        self.tpr_list = tpr_list
        self.auc_list = auc_list
        self.cm_list = cm_list

        self.auc = np.mean(auc_vals)
        self.acc = np.mean(acc_vals)
        self.prec = np.mean(prec_vals)
        self.recall = np.mean(recall_vals)
        self.f1 = np.mean(f1_vals)

    def get_result(self) -> dict:
        return {
            "accuracy": round(self.acc, 3),
            "precision": round(self.prec, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
        }

    def plot_roc(self) -> None:
        plt.figure()
        for fpr, tpr, roc_auc in zip(self.fpr_list, self.tpr_list, self.auc_list):
            plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()

    def plot_cm(self) -> None:
        cm = np.sum(self.cm_list, axis=0)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()