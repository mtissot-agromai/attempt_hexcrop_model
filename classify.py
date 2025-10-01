import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np
import random
import os, sys

global results

results = {
    "bean": [],
    "wheat": [],
    "soybean": [],
    "maize": [],
    "rice": [],
    "total": [],
    "seed1": [],
    "seed2": []
}

def classify_cultures(df: pd.DataFrame, features_list: list, target_column: str = 'culture', test_size_ratio: float = 0.10, run: int = 0):
    """
    Trains the moedl to predict culture. Ideally this would be 
    """
    if not all(col in df.columns for col in features_list + [target_column]):
        missing = [col for col in features_list + [target_column] if col not in df.columns]
        print(f"Error: feature col not in CSV file: {missing}")
        return

    X = df[features_list]
    y = df[target_column]
    
    data_points_before = len(X)
    valid_indices = X.dropna().index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    data_points_after = len(X)
    if data_points_before != data_points_after:
        print(f"Warning: Dropped {data_points_before - data_points_after} rows with missing feature values.")
        

    r_state1 = int(1_000_000 * random.random())
    # r_state1 = 345007 # melhor resultado

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size_ratio, 
        random_state=r_state1,
        stratify=y
    )
    
    print(f"\n--- Data Split Summary ---")
    print(f"Total samples: {len(X)}")
    print(f"Training samples ({100-test_size_ratio*100:.0f}%): {len(X_train)}")
    print(f"Testing samples ({test_size_ratio*100:.0f}%): {len(X_test)}")
    print(f"Test Set Culture Distribution:\n{y_test.value_counts(normalize=True).round(2)}")

    r_state2 = int(1_000_000 * random.random())
    # r_state2 = 382647 # melhor resultado

    model = RandomForestClassifier(n_estimators=100, random_state=r_state2)
    model.fit(X_train, y_train)

    # Salvar modelo vvvvvv

    # initial_type = [('input', FloatTensorType([None, X_train.shape[1]]))]

    # onnx_model = convert_sklearn(model, initial_types=initial_type)

    # with open("hexcrop_model.onnx", "wb") as f:
    #     f.write(onnx_model.SerializeToString())

    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    class_labels = model.classes_

    probabilities_df = pd.DataFrame(probabilities, columns=[f"Prob_{c}" for c in class_labels])
    probabilities_df['Predicted_Class'] = y_pred
    probabilities_df['Actual_Class'] = y_test.reset_index(drop=True)

    probabilities_df.to_csv("probas.csv")

    print(f"Y predicted: {y_pred}")
    
    print("\n--- Model Performance on Test Set ---")

    print(f"\nRandom seed1: {r_state1}\n")
    print(f"\nRandom seed2: {r_state2}\n")
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"classification Accu total: {accuracy:.2f}\n")
    
    test_results = X_test.copy()
    test_results['Actual_Culture'] = y_test
    test_results['Predicted_Culture'] = y_pred
    test_results['Result'] = (test_results['Actual_Culture'] == test_results['Predicted_Culture']).astype(int)


    os.makedirs(f"./results/{r_state1}", exist_ok=True)
    results["bean"].append(test_results[test_results['Actual_Culture']=='bean']['Result'].mean())
    results["wheat"].append(test_results[test_results['Actual_Culture']=='wheat']['Result'].mean())
    results["soybean"].append(test_results[test_results['Actual_Culture']=='soybean']['Result'].mean())
    results["maize"].append(test_results[test_results['Actual_Culture']=='maize']['Result'].mean())
    results["rice"].append(test_results[test_results['Actual_Culture']=='rice']['Result'].mean())
    results["total"].append(test_results['Result'].mean())
    results["seed1"].append(r_state1)
    results["seed2"].append(r_state2)

    print("\n--- 5 Example Test Predictions ---")
    test_results.to_csv(f"results/{r_state1}/test_results_{run}.csv")

FEATURES = [
               'B4_value_at_25pct','B4_median','B4_mean','B4_q25','B4_q75',
               'B5_value_at_25pct', 'B5_q25', 'B5_median', 'B5_mean', 'B5_min',
               'B11_q25','B12_mean','B11_median','B12_q75','B11_mean','B11_value_at_75pct','B11_min','B11_value_at_25pct','B11_q75','B11_value_at_50pct',
               'B12_median','B12_skewness','B12_value_at_75pct','B12_value_at_25pct','B12_q25','B12_kurtosis','B12_min','B12_value_at_50pct'
               ]

bean_df = pd.read_csv("features_cultures/bean_features.csv")
soybean_df = pd.read_csv("features_cultures/soybean_features.csv")
maize_df = pd.read_csv("features_cultures/maize_features.csv")
wheat_df = pd.read_csv("features_cultures/wheat_features.csv")
rice_df = pd.read_csv("features_cultures/rice_features.csv")

df = pd.concat([soybean_df, bean_df, maize_df, rice_df, wheat_df])

for i in range(100):
    runn = i+1
    classify_cultures(
        df=df, 
        features_list=FEATURES, 
        target_column='culture',
        test_size_ratio=0.10,
        run=runn
    )

print(results)

results_df = pd.DataFrame(results)

results_df.to_csv(f"results/final_results.csv")