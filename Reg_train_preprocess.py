
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import os
import warnings
warnings.filterwarnings("ignore")
#python3 pipeline.py
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
            X = X.copy()

            X = X.fillna(0)

            X['skill_score'] = (X['technical_skill_score'] + X['soft_skill_score']) / 2

            X['experience'] = X['internship_count'] + X['live_projects'] + X['certifications']+ (X['work_experience_months']/12)

            X['academic_score'] = (
                X['ssc_percentage'] + X['hsc_percentage'] +
                X['degree_percentage'] +(10 * X['cgpa'])) / 4

            X['progress'] = X['degree_percentage'] - X['ssc_percentage']

            X['cgpa_square'] = X['cgpa'] ** 2

            X['total_experience'] = X['experience'] * X['academic_score']

            X['academic_skill'] = X['academic_score'] * X['technical_skill_score']

            X['projects_skill'] = X['live_projects'] * X['technical_skill_score']
            
            X['skill_backlogs']= X['skill_score'] - (X['backlogs']*10)
            
            X['hire_score']= X['skill_score'] * 0.5 +X['experience'] * 0.2 +X['academic_score'] * 0.3

            raw_cols_to_drop = ['technical_skill_score', 'soft_skill_score', 'ssc_percentage',
                                'hsc_percentage','degree_percentage','cgpa','internship_count',
                                'live_projects','certifications','work_experience_months','gender',
                                'extracurricular_activities']

            X = X.drop(columns=[c for c in raw_cols_to_drop if c in X.columns])

            return X
        

def train_model(x_train, y_train):
    os.makedirs("artifacts", exist_ok=True)

    num_feat = ['entrance_exam_score', 'attendance_percentage', 'skill_score',
                'academic_score', 'progress', 'skill_backlogs', 'hire_score',
                'cgpa_square', 'total_experience', 'academic_skill', 'projects_skill']


    placement_pred = Pipeline([
        ('feature_engineering', FeatureEngineer()),
        ('preprocessing', ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_feat)
        ], remainder='drop')),
        ('regressor', RandomForestRegressor(random_state=42,n_jobs=-1))])
                                    
    mlflow.set_experiment("Student Placement Prediction")

    with mlflow.start_run() as run:

        placement_pred.fit(x_train, y_train)

        joblib.dump(placement_pred, "artifacts/Salary_prediction_pipeline.pkl")
        mlflow.sklearn.log_model(placement_pred,name="model",input_example=x_train.iloc[:3])

    return run.info.run_id