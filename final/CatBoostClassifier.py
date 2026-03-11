# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: ML_challenge
@File Name:    CatBoostClassifier.py
@Software:     Python
@Time:         May/2026
@Author:       Rui Xu
@Description:  Optimized classification using LightGBM + TargetEncoding + KFold CV.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")


class CatBoostPipeline:
    def __init__(self, train_path, eval_path):
        self.train_path = train_path
        self.eval_path = eval_path
        self.label_encoder = LabelEncoder()
        # Separate encoders for categorical features to handle strings before SMOTE
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
        # 1. Use 'species' as the correct target column
        X = self.train_data.drop(columns=['species'])
        y = self.label_encoder.fit_transform(self.train_data['species'])

        # 2. Identify feature types
        self.cat_feature_names = X.select_dtypes(include=['object', 'bool']).columns.tolist()
        self.numerical_feature_names = X.select_dtypes(exclude=['object', 'bool']).columns.tolist()

        # 3. Handle Categorical features: Must encode them for SMOTE to work
        for col in self.cat_feature_names:
            le = LabelEncoder()
            # Combine train and eval to ensure all labels are captured
            combined_data = pd.concat([X[col], self.eval_data[col]], axis=0).astype(str)
            le.fit(combined_data)
            X[col] = le.transform(X[col].astype(str))
            self.feature_encoders[col] = le

        # 4. Handle numerical missing values
        X[self.numerical_feature_names] = X[self.numerical_feature_names].fillna(X[self.numerical_feature_names].mean())

        # 5. Scale numerical features
        X[self.numerical_feature_names] = self.scaler.fit_transform(X[self.numerical_feature_names])

        # 6. Process evaluation set using the same logic
        X_eval = self.eval_data.copy()
        for col in self.cat_feature_names:
            X_eval[col] = self.feature_encoders[col].transform(X_eval[col].astype(str))

        X_eval[self.numerical_feature_names] = X_eval[self.numerical_feature_names].fillna(
            X_eval[self.numerical_feature_names].mean())
        X_eval[self.numerical_feature_names] = self.scaler.transform(X_eval[self.numerical_feature_names])
        self.X_eval_processed = X_eval

        return X, y

    def run_cv_training(self, X, y):
        # Using 3-split as per your original code
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        best_model = None
        best_acc = 0

        # Calculate class weights for imbalance
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) == 2:
            class_weights = [1.0] * len(unique_classes)
            min_class_idx = np.argmin(class_counts)
            max_class_idx = np.argmax(class_counts)
            class_weights[min_class_idx] = class_counts[max_class_idx] / class_counts[min_class_idx]
        else:
            class_weights = None

        param_distributions = {
            'iterations': [800, 1200],
            'learning_rate': [0.03, 0.05],
            'depth': [6, 8],
            'l2_leaf_reg': [1, 3],
            'subsample': [0.8, 0.9],
            'bootstrap_type': ['Bernoulli']
        }

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Apply SMOTE - This now works because cat features are label-encoded
            sm = SMOTE(random_state=42)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

            base_model = CatBoostClassifier(
                verbose=0,
                random_seed=42,
                eval_metric='Accuracy',
                early_stopping_rounds=100,
                task_type='CPU',  # Changed to CPU for compatibility, change to GPU if available
                class_weights=class_weights
            )

            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=10,
                cv=2,
                scoring='accuracy',
                n_jobs=1,
                verbose=0,
                random_state=42
            )

            random_search.fit(X_train_res, y_train_res)

            print(f"[{fold + 1}] Best parameters found: {random_search.best_params_}")

            model = random_search.best_estimator_

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"[Fold {fold + 1}] Accuracy: {acc * 100:.2f}%")
            scores.append(acc)

            # Print detailed classification report
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

            if acc > best_acc:
                best_acc = acc
                best_model = model

        print(f"\nAverage CV Accuracy: {np.mean(scores) * 100:.2f}%")
        self.model = best_model

    def predict_eval_set(self):
        preds = self.model.predict(self.X_eval_processed)
        # Handle 2D output from CatBoost if necessary
        if len(preds.shape) > 1:
            preds = preds.flatten()
        labels = self.label_encoder.inverse_transform(preds.astype(int))
        np.savetxt("predicted_labels_catboost.txt", labels, fmt="%s")
        print("Saved predictions to 'predicted_labels_catboost.txt'")

    def run(self):
        self.load_data()
        X, y = self.preprocess()
        self.run_cv_training(X, y)
        self.predict_eval_set()


if __name__ == "__main__":
    pipeline = CatBoostPipeline(
        "TrainOnMe_orig.csv",
        "EvaluateOnMe.csv"
    )
    pipeline.run()