#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project:     ML_challenge
@File:        main_randomforest_pipeline.py
@Author:      Rui Xu
@Time:        May/2026
@Description: Classification pipeline using RandomForest + SMOTE + KFold CV.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


class RandomForestPipeline:

    def __init__(self, train_path, eval_path):
        self.train_path = train_path
        self.eval_path = eval_path
        self.label_encoder = LabelEncoder()
        self.feature_encoders = {}
        self.cat_feature_names = None
        self.numerical_feature_names = None
        self.scaler = StandardScaler()
        self.model = None
        self.X_eval_processed = None


    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.eval_data = pd.read_csv(self.eval_path)

    def preprocess(self):
        X = self.train_data.drop(columns=['species'])
        y = self.label_encoder.fit_transform(self.train_data['species'])
        # Identify feature types
        self.cat_feature_names = X.select_dtypes(
            include=['object', 'bool']
        ).columns.tolist()
        self.numerical_feature_names = X.select_dtypes(
            exclude=['object', 'bool']
        ).columns.tolist()

        # Encode categorical features
        for col in self.cat_feature_names:
            le = LabelEncoder()
            combined_data = pd.concat(
                [X[col], self.eval_data[col]],
                axis=0
            ).astype(str)
            le.fit(combined_data)
            X[col] = le.transform(X[col].astype(str))
            self.feature_encoders[col] = le

        # Fill numerical missing values
        X[self.numerical_feature_names] = \
            X[self.numerical_feature_names].fillna(
                X[self.numerical_feature_names].mean()
            )

        # Scale numerical features
        X[self.numerical_feature_names] = \
            self.scaler.fit_transform(
                X[self.numerical_feature_names]
            )

        # Process evaluation set
        X_eval = self.eval_data.copy()
        for col in self.cat_feature_names:
            X_eval[col] = self.feature_encoders[col].transform(
                X_eval[col].astype(str)
            )
        X_eval[self.numerical_feature_names] = \
            X_eval[self.numerical_feature_names].fillna(
                X_eval[self.numerical_feature_names].mean()
            )

        X_eval[self.numerical_feature_names] = \
            self.scaler.transform(
                X_eval[self.numerical_feature_names]
            )

        self.X_eval_processed = X_eval
        return X, y


    def run_cv_training(self, X, y):
        skf = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42
        )

        scores = []
        best_model = None
        best_acc = 0

        # class imbalance weights
        unique_classes, class_counts = np.unique(
            y,
            return_counts=True
        )

        if len(unique_classes) == 2:

            class_weights = {
                unique_classes[i]:
                class_counts.max() / class_counts[i]
                for i in range(len(unique_classes))
            }

        else:
            class_weights = None

        param_distributions = {
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        }

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            # SMOTE
            sm = SMOTE(random_state=42)
            X_train_res, y_train_res = sm.fit_resample(
                X_train,
                y_train
            )

            base_model = RandomForestClassifier(

                random_state=42,
                class_weight=class_weights,
                n_jobs=-1

            )

            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=15,
                cv=2,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )

            random_search.fit(
                X_train_res,
                y_train_res
            )
            print(f"[Fold {fold+1}] Best Params: {random_search.best_params_}")
            model = random_search.best_estimator_
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"[Fold {fold+1}] Accuracy: {acc*100:.2f}%")
            print(
                classification_report(
                    y_test,
                    y_pred,
                    target_names=self.label_encoder.classes_
                )
            )
            scores.append(acc)

            if acc > best_acc:
                best_acc = acc
                best_model = model

        print("\nAverage CV Accuracy:", np.mean(scores)*100)
        self.model = best_model

    def predict_eval_set(self):
        preds = self.model.predict(self.X_eval_processed)
        labels = self.label_encoder.inverse_transform(
            preds.astype(int)
        )

        np.savetxt(
            "predicted_labels_randomforest.txt",
            labels,
            fmt="%s"
        )

        print("Saved predictions to predicted_labels_randomforest.txt")

    def run(self):
        self.load_data()
        X, y = self.preprocess()
        self.run_cv_training(X, y)
        self.predict_eval_set()


if __name__ == "__main__":
    pipeline = RandomForestPipeline(
        "TrainOnMe_orig.csv",
        "EvaluateOnMe.csv"
    )
    pipeline.run()