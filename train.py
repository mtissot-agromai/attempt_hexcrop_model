import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import numpy as np
import sys, os, argparse

from itertools import combinations

import logging, sys
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

error_file_handler = logging.FileHandler('errors.log')
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(formatter)

logger.handlers = []
logger.addHandler(stdout_handler)
logger.addHandler(error_file_handler)

def save_feature_importance(pipeline: Pipeline, preprocessor: ColumnTransformer, PATH: str, FILENAME: str):
    final_classifier = pipeline.named_steps["clf"]
    importances_final = final_classifier.feature_importances_

    preprocessor = pipeline.named_steps["feature_selection_forced"]

    output_features = preprocessor.get_feature_names_out()

    selected_feature_names = [f.split('__')[-1] for f in output_features]

    feature_importances_series = pd.Series(importances_final, index=selected_feature_names)

    top_features = feature_importances_series.sort_values(ascending=False)

    top_features.to_csv(f"{PATH}/{FILENAME}", index=True, header=['Importance'], index_label='Feature')

def train_single_model(df: pd.DataFrame, cult1: str, cult2: str, model_path: str, save_csv: bool = True):
    logger.info(f"Beginning training of single model for cultures {cult1} and {cult2}")
    df_cult = df.copy()
    df_cult = df[(df['culture']==cult1) | (df['culture']==cult2)]
    X = df_cult.drop(columns=["filename", "culture", "year", "season"])
    y = df_cult['culture']

    model_name = f"{cult1}_{cult2}.onnx"

    r_state = 264284 # escolhe oq quiser, mas grava o valor p manter reproducibilidade

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=r_state
    )

    all_features = X_train.columns.tolist()

    features_for_selection_indices = list(range(1, len(all_features))) 

    logger.info(f"Selecting best features")

    max_features = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    # 84.5, 83.6, 84.5, 83.7, 85.3, 85.7, 84.4, 83.6, 84.4, 84.0, 84.48

    feature_selector = SelectFromModel(
        estimator=RandomForestClassifier(n_estimators=150, random_state=r_state),
        max_features=max_features[5]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("force_feature", "passthrough", [0]), 
            
            ("select_features", feature_selector, features_for_selection_indices)
        ]
    )

    pipeline = Pipeline([
        ("feature_selection_forced", preprocessor),  
        ("clf", RandomForestClassifier(n_estimators=250, random_state=r_state))
    ])

    logger.info(f"Fitting the data")

    pipeline.fit(X_train, y_train) 

    save_feature_importance(pipeline, preprocessor, model_path, f"{cult1}_{cult2}_feature_importance.csv")

    initial_type = [('input', FloatTensorType([None, X_train.shape[1]]))]

    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

    if model_path:
        os.makedirs(model_path, exist_ok=True)
        with open(f"{model_path}/{model_name}", "wb") as f:
            f.write(onnx_model.SerializeToString())

    logger.info(f"Saved model to {model_path}/{model_name}")

    y_pred = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)

    model_classes = pipeline.classes_

    results_df = pd.DataFrame(index=X_test.index)
    results_df['Actual_Culture'] = y_test
    results_df['Predicted_Culture'] = y_pred
    results_df['Correct'] = (y_test == y_pred).astype(int)

    cultures_list = model_classes

    top_2_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :2]
    top_2_cultures = np.array(model_classes)[top_2_indices]
    actual_culture_array = y_test.values.reshape(-1, 1)
    results_df['Correct2'] = np.any(top_2_cultures == actual_culture_array, axis=1).astype(int)

    prob_cols = [cultures_list[i] for i in range(len(probabilities[0]))]

    probabilities_df = pd.DataFrame(probabilities,columns=prob_cols,index=X_test.index)

    final_df = results_df.join(probabilities_df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r_state)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    logger.info(f"Accuracy per fold: {scores}")
    logger.info(f"Average accuracy: {scores.mean()}")
    print("\n\n")

    final_df['Avg Accuracy'] = scores.mean()

    if save_csv:
        os.makedirs(f"{model_path}/csvs/", exist_ok=True)
        final_df.to_csv(f"{model_path}/csvs/check_{cult1}_{cult2}.csv", index=False)
        logger.info(f"Saved CSV to {model_path}/csvs/check_{cult1}_{cult2}.csv")

def train_general_model(df: pd.DataFrame, model_path: str, save_csv: bool = True):
    logger.info(f"Beginning training of general model")
    df_cult = df.copy()
    X = df_cult.drop(columns=["filename", "culture", "year", "season"])
    y = df_cult['culture']

    model_name = f"general_model.onnx"

    r_state = 264284 # escolhe oq quiser, mas grava o valor p manter reproducibilidade

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=r_state
    )

    all_features = X_train.columns.tolist()

    features_for_selection_indices = list(range(0, len(all_features))) 

    feature_selector = SelectFromModel(
        estimator=RandomForestClassifier(n_estimators=100, random_state=r_state),
        max_features=100 
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("select_features", feature_selector, features_for_selection_indices)
        ]
    )

    pipeline = Pipeline([
        ("feature_selection_forced", preprocessor),  
        ("clf", RandomForestClassifier(n_estimators=150, random_state=r_state))
    ])

    pipeline.fit(X_train, y_train) 

    save_feature_importance(pipeline, preprocessor, model_path, f"general_feature_importance.csv")

    initial_type = [('input', FloatTensorType([None, X_train.shape[1]]))]

    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

    if model_path:
        os.makedirs(model_path, exist_ok=True)
        with open(f"{model_path}/{model_name}", "wb") as f:
            f.write(onnx_model.SerializeToString())
    logger.info(f"Saved model to {model_path}/{model_name}")

    y_pred = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)

    model_classes = pipeline.classes_

    results_df = pd.DataFrame(index=X_test.index)
    results_df['Actual_Culture'] = y_test
    results_df['Predicted_Culture'] = y_pred
    results_df['Correct'] = (y_test == y_pred).astype(int)

    cultures_list = model_classes

    top_2_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :2]
    top_2_cultures = np.array(model_classes)[top_2_indices]
    actual_culture_array = y_test.values.reshape(-1, 1)
    results_df['Correct2'] = np.any(top_2_cultures == actual_culture_array, axis=1).astype(int)

    prob_cols = [cultures_list[i] for i in range(len(probabilities[0]))]

    probabilities_df = pd.DataFrame(probabilities,columns=prob_cols,index=X_test.index)

    final_df = results_df.join(probabilities_df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r_state)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    final_df['Avg Accuracy'] = scores.mean()

    if save_csv:
        os.makedirs(f"{model_path}/csvs/", exist_ok=True)
        final_df.to_csv(f"{model_path}/csvs/check_general.csv", index=False)
        logger.info(f"Saved CSV to {model_path}/csvs/check_general.csv")

def main(args):
    parser = argparse.ArgumentParser(description="Training Hexcrop models.")

    parser.add_argument("--unique", type=str, nargs=2, help="If you want to train a unique model at a time")

    parser.add_argument("--general",
                        action='store_true',
                        dest='TRAIN_GENERAL_MODEL',
                        default=False,
                        help="If you want to train the general model")

    parser.add_argument("--doubles",
                        action='store_true',
                        dest='TRAIN_ALL_CULTURES',
                        default=False,
                        help="If you want to train all 2-culture models")

    parser.add_argument("--nocsv", 
                        action='store_false',
                        dest='SAVE_CSV',
                        default=True,
                        help='Do not save the CSV checking file')
    
    parser.add_argument("--output", nargs=1, type=str, help='The path to save the models.')

    parser.add_argument("--input", nargs=1, type=str, help='The path to the training dataset.')

    args = parser.parse_args()

    GENERAL = args.TRAIN_GENERAL_MODEL
    TRAIN_ALL_CULTURES = args.TRAIN_ALL_CULTURES
    SAVE_CSV = args.SAVE_CSV

    OUTPUT_PATH='models'

    UNIQUE_CULTURES = []

    INPUT_PATH = "."

    if args.unique:
        lower = [un.lower() for un in args.unique]
        UNIQUE_CULTURES = [x.lower() for x in sorted(lower)]
        GENERAL=False
        TRAIN_ALL_CULTURES=False

    if args.output:
        OUTPUT_PATH=args.output[0]

    if args.input:
        INPUT_PATH = args.input[0]

    PARENT_PATH = '.'

    features_df = pd.read_csv(f"{INPUT_PATH}/training_dataset.csv")

    # ! OPCIONAL: DROPAR TRIGO
    if 'wheat' in list(features_df['culture'].unique()):
        features_df = features_df[features_df['culture'] != 'wheat']
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    CULTURES = sorted(list(features_df['culture'].unique()))

    if UNIQUE_CULTURES:
        for cult in UNIQUE_CULTURES:
            if cult not in CULTURES:
                logger.error(f"You entered culture {cult}, which does not exist on dataframe['cultures'].")
                sys.exit(1)
        print("\n====================")
        culture_combos = list(combinations(UNIQUE_CULTURES, 2))
        for cult1, cult2 in culture_combos:
            train_single_model(features_df, cult1, cult2, f"{PARENT_PATH}/{OUTPUT_PATH}", save_csv=SAVE_CSV)

    if TRAIN_ALL_CULTURES:
        print("\n====================")
        culture_combos = list(combinations(CULTURES, 2))
        for cult1, cult2 in culture_combos:
            train_single_model(features_df, cult1, cult2, f"{PARENT_PATH}/{OUTPUT_PATH}", save_csv=SAVE_CSV)

    if GENERAL:
        print("\n====================")
        train_general_model(features_df, f"{PARENT_PATH}/{OUTPUT_PATH}", save_csv=SAVE_CSV)
    
    pass

if __name__ == "__main__":
    main(sys.argv)

