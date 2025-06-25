import os, sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys) # type: ignore
    
def load_object(file_path: str) -> object:
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) # type: ignore

def evaluate_model(X_train, y_train, X_test, y_test, models: dict, param: dict) -> dict:
    try:
        report = {}
        for model_name, model in models.items():
            print(f"Training model {model_name}")
            para=param[model_name]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_r2_score = r2_score(y_train, y_train_pred)
            test_r2_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_r2_score
        return report
    except Exception as e:
        raise CustomException(e, sys) # type: ignore
