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

    feature_selector = SelectFromModel(
        estimator=RandomForestClassifier(n_estimators=100, random_state=r_state),
        max_features=100 
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("force_feature", "passthrough", [0]), 
            
            ("select_features", feature_selector, features_for_selection_indices)
        ]
    )

    pipeline_forced = Pipeline([
        ("feature_selection_forced", preprocessor),  
        ("clf", RandomForestClassifier(n_estimators=150, random_state=r_state))
    ])

    logger.info(f"Fitting the data")

    pipeline_forced.fit(X_train, y_train) 

    initial_type = [('input', FloatTensorType([None, X_train.shape[1]]))]

    onnx_model = convert_sklearn(pipeline_forced, initial_types=initial_type)

    if model_path:
        os.makedirs(model_path, exist_ok=True)
        with open(f"{model_path}/{model_name}", "wb") as f:
            f.write(onnx_model.SerializeToString())

    logger.info(f"Saved model to {model_path}/{model_name}")

    y_pred = pipeline_forced.predict(X_test)

    probabilities = pipeline_forced.predict_proba(X_test)

    model_classes = pipeline_forced.classes_

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
    scores = cross_val_score(pipeline_forced, X, y, cv=cv, scoring="accuracy")

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
        ("feature_selection", preprocessor),  
        ("clf", RandomForestClassifier(n_estimators=150, random_state=r_state))
    ])

    pipeline.fit(X_train, y_train) 

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

    args = parser.parse_args()

    GENERAL = args.TRAIN_GENERAL_MODEL
    TRAIN_ALL_CULTURES = args.TRAIN_ALL_CULTURES
    SAVE_CSV = args.SAVE_CSV

    OUTPUT_PATH='models'

    UNIQUE_CULTURES = []

    if args.unique:
        UNIQUE_CULTURES = [x.lower() for x in sorted(args.unique)]
        GENERAL=False
        TRAIN_ALL_CULTURES=False

    if args.output:
        OUTPUT_PATH=args.output[0]

    PARENT_PATH = '.'

    features_df = pd.read_csv(f"{PARENT_PATH}/training_dataset.csv")

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

