import onnxruntime as rt
import numpy as np
import pandas as pd
import os

ONNX_MODEL_PATH = "hexcrop_model_new.onnx"

INPUT_CSV_PATH = "tapera_limited_features.csv"

OUTPUT_CSV_PATH = "predictions_output.csv"

MODEL_INPUT_NAMES = [
    'B4_value_at_25pct','B4_median','B4_mean','B4_q25','B4_q75',
    'B5_value_at_25pct', 'B5_q25', 'B5_median', 'B5_mean', 'B5_min',
    'B11_q25','B12_mean','B11_median','B12_q75','B11_mean','B11_value_at_75pct','B11_min','B11_value_at_25pct','B11_q75','B11_value_at_50pct',
    'B12_median','B12_skewness','B12_value_at_75pct','B12_value_at_25pct','B12_q25','B12_kurtosis','B12_min','B12_value_at_50pct'
]

def batch_predict_with_onnx(model_path: str, input_csv_path: str, output_csv_path: str, feature_names: list, id_column: str):
    if not os.path.exists(input_csv_path):
        print(f"Error: Input file not found at {input_csv_path}")
        return

    try:
        df_input = pd.read_csv(input_csv_path)
        
        req_cols = [id_column] + feature_names # fdentificador + faetures
        missing_cols = [col for col in req_cols if col not in df_input.columns]
        
        if missing_cols:
            print(f"Error: Missing columns in input CSV: {missing_cols}")
            return

        X_test = df_input[feature_names]
        
    except Exception as e:
        print(f"Error loading or processing input CSV: {e}")
        return

    try:
        sess = rt.InferenceSession(model_path)
    except Exception as e:
        print(f"Error loading ONNX model at {model_path}: {e}")
        return

    model_input_name = sess.get_inputs()[0].name
    model_input_type = np.float32 # pq salvei usando float32 la no classify.py
    output_names = [output.name for output in sess.get_outputs()]
    
    print(f"Successfully loaded model. Predicting {len(df_input)} cases...")

    input_array = X_test.values.astype(model_input_type)
    input_feed = {model_input_name: input_array}

    try:
        raw_outputs = sess.run(output_names, input_feed)
    except Exception as e:
        print(f"Error during ONNX inference: {e}")
        return

    predictions = raw_outputs[0]
    probabilities = raw_outputs[1]
    
    df_results = pd.DataFrame()
    
    df_results[id_column] = df_input[id_column]
    

    df_results['prediction'] = predictions
    
    print(probabilities[0])
    df_prob = pd.DataFrame(probabilities, columns=list(probabilities[0].keys()))
    df_results = pd.concat([df_results, df_prob], axis=1)

    df_results.to_csv(output_csv_path, index=False)

    print(f"csv prediction cmoplete for csv {input_csv_path}!")
    print(f"Results saved to: {output_csv_path}")

# --- Execution ---

if __name__ == "__main__":

    # Run the main function
    batch_predict_with_onnx(
        model_path=ONNX_MODEL_PATH,
        input_csv_path=INPUT_CSV_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        feature_names=MODEL_INPUT_NAMES,
        id_column="filename"
    )