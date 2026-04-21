
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import os
import warnings
warnings.filterwarnings("ignore")
#python3 Class_pipeline.py
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
            X = X.copy()

            X = X.fillna(0)

            # basic engineered features
            X['skill_score'] = (X['technical_skill_score'] + X['soft_skill_score']) / 2

            X['academic_score'] = (
                X['ssc_percentage'] + X['hsc_percentage'] +
                X['degree_percentage'] +(10 * X['cgpa'])) / 4

            X['experience'] = (
                X['internship_count'] +X['live_projects'] +
                X['certifications'] +(X['work_experience_months'] / 12)
            )

            X['Skill by work duration'] = X['skill_score'] * X['work_experience_months']

            X['Skill by total experience'] = X['skill_score'] * X['experience']

            X['progress'] = X['degree_percentage'] - X['ssc_percentage']

            X['cgpa_square'] = X['cgpa'] ** 2

            X['total_experience'] = X['experience'] * X['academic_score']

            X['academic_skill'] = X['academic_score'] * X['technical_skill_score']

            X['projects_skill'] = X['live_projects'] * X['technical_skill_score']

            X['gender'] = X['gender'].map({'male': 1, 'female': 0})

            X['extracurricular_activities'] = X['extracurricular_activities'].map({'Yes': 1,'No': 0})

            raw_cols_to_drop = [
                'technical_skill_score',
                'soft_skill_score',
                'ssc_percentage',
                'hsc_percentage',
                'degree_percentage',
                'cgpa',
                'internship_count',
                'live_projects',
                'certifications',
                'work_experience_months',
                'entrance_exam_score'
            ]

            X = X.drop(columns=[c for c in raw_cols_to_drop if c in X.columns])


            return X
        

def train_model(x_train, y_train):
    os.makedirs("artifacts", exist_ok=True)

    num_feat = ['skill_score', 'academic_score', 'experience',
                'Skill by work duration', 'Skill by total experience',
                'progress', 'cgpa_square', 'total_experience',
                'academic_skill', 'projects_skill']


    placement_pred = Pipeline([
        ('feature_engineering', FeatureEngineer()),
        ('preprocessing', ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_feat)
        ], remainder='drop')),
        ('classifier', DecisionTreeClassifier(max_depth=5,min_samples_leaf=5,random_state=42))])
                                    
    mlflow.set_experiment("Student Placement Prediction")

    with mlflow.start_run() as run:

        placement_pred.fit(x_train, y_train)

        joblib.dump(placement_pred, "artifacts/Placement_prediction_pipeline.pkl")
        mlflow.sklearn.log_model(placement_pred,name="model",input_example=x_train.iloc[:3])

    return run.info.run_id