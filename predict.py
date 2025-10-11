import pandas as pd
import numpy as np
import onnxruntime as rt
from itertools import combinations
from typing import Dict, Tuple, List, Any

import logging, sys, os, argparse
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

def get_onnx_input_name(session: rt.InferenceSession) -> str:
    return session.get_inputs()[0].name

def create_pairwise_model_map(cultures: List[str], models_path: str) -> Dict[Tuple[str, str], str]:
    model_map = {}
    
    for cult1, cult2 in combinations(sorted(cultures), 2):
        filename = f"{cult1}_{cult2}.onnx"
        
        model_map[(cult1, cult2)] = filename
        model_map[(cult2, cult1)] = filename
        
    return model_map

def predict_single_case(
    session: rt.InferenceSession, 
    features: List[str], 
    data: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:

    ordered_values = [data[name] for name in features]
    input_array = np.array([ordered_values], dtype=np.float32)

    input_name = get_onnx_input_name(session)
    input_feed = {input_name: input_array}

    return session.run(None, input_feed)

def run_cascading_prediction(
    new_data_df: pd.DataFrame,
    models_path: str,
    full_feature_list: List[str],
    class_list: List[str],
    PAIRWISE_MODEL_MAP: List[Tuple[str, str]]
) -> pd.DataFrame:
    
    if not new_data_df.empty:
        new_data = new_data_df.copy()
    else:
         logger.error(f"Dataframe is empty. Exiting now")
         sys.exit(1)

    try:
        general_sess = rt.InferenceSession(f"{models_path}/general_model.onnx")
    except Exception as e:
        logger.error(f"Error loading general model: {e}")
        return pd.DataFrame()

    final_predictions = []

    for index, sample_data in new_data.iterrows():
        _, general_probs_list = predict_single_case(general_sess, full_feature_list, sample_data)
        
        general_probs_dict = general_probs_list[0]

        general_probs = np.array([general_probs_dict[c] for c in class_list])

        top_2_indices = np.argsort(general_probs)[::-1][:2]
        
        cult_A = class_list[top_2_indices[0]]
        cult_B = class_list[top_2_indices[1]]

        prob_A_general = general_probs_dict[cult_A]
        prob_B_general = general_probs_dict[cult_B]
        
        try:
            model_file = PAIRWISE_MODEL_MAP[(cult_A, cult_B)]
        except KeyError:
            final_pred = cult_A 
            final_prob = general_probs[top_2_indices[0]]
            pairwise_model_used = 'Error_Fallback'
            prob_A_pair, prob_B_pair = np.nan, np.nan
            logger.error(f"Error: Model not defined for pair ({cult_A}, {cult_B})")
            continue

        try:
            pairwise_sess = rt.InferenceSession(f"{models_path}/{model_file}")
        except Exception:
            logger.warning(f"Warning: Could not load pairwise model {models_path}/{model_file}. Skipping sample.")
            final_pred = cult_A
            final_prob = general_probs[top_2_indices[0]]
            pairwise_model_used = f'{model_file}_LoadError'
            prob_A_pair, prob_B_pair = np.nan, np.nan
            
        else:
            _, pairwise_probs_raw = predict_single_case(pairwise_sess, full_feature_list, sample_data)
            
            pairwise_probs_dict = pairwise_probs_raw[0]

            prob_A_value = pairwise_probs_dict[cult_A]
            prob_B_value = pairwise_probs_dict[cult_B]

            pairwise_model_used = model_file
            prob_A_pair = prob_A_value
            prob_B_pair = prob_B_value

            weighted_score_A = prob_A_pair
            weighted_score_B = prob_B_pair
            
            if weighted_score_A > weighted_score_B:
                final_pred = cult_A
                final_prob = weighted_score_A
            else:
                final_pred = cult_B
                final_prob = weighted_score_B
        
        final_predictions.append({
            'Index': index,
            'Top_1_General': cult_A,
            'Top_2_General': cult_B,
            f'Prob_General_{cult_A}': prob_A_general,
            f'Prob_General_{cult_B}': prob_B_general,
            
            'Pairwise_Model_Used': pairwise_model_used,
            f'Prob_Pairwise_{cult_A}': prob_A_pair,
            f'Prob_Pairwise_{cult_B}': prob_B_pair,
            
            'Final_Prediction': final_pred,
            'Final_Confidence': final_prob
        })

    return pd.DataFrame(final_predictions).set_index('Index')

def run_binary_prediction(
    new_data_df: pd.DataFrame,
    models_path: str,
    full_feature_list: List[str],
    class_list: List[str],
    PAIRWISE_MODEL_MAP: List[Tuple[str, str]],
) -> pd.DataFrame:
    
    if not new_data_df.empty:
        new_data = new_data_df.copy()
    else:
         logger.error(f"Dataframe is empty. Exiting now")
         sys.exit(1)

    cult_A = class_list[0]
    cult_B = class_list[1]

    model_file = PAIRWISE_MODEL_MAP[(cult_A, cult_B)]

    try:
        session = rt.InferenceSession(f"{models_path}/{model_file}")
    except Exception as e:
        logger.error(f"Error loading general model: {e}")
        return pd.DataFrame()

    final_predictions = []

    for index, sample_data in new_data.iterrows():
        _, general_probs_list = predict_single_case(session, full_feature_list, sample_data)
        
        general_probs_dict = general_probs_list[0]

        general_probs = np.array([general_probs_dict[c] for c in class_list])

        top_2_indices = np.argsort(general_probs)[::-1][:2]
        
        cult_A = class_list[top_2_indices[0]]
        cult_B = class_list[top_2_indices[1]]

        prob_A_culture = general_probs_dict[cult_A]
        prob_B_culture = general_probs_dict[cult_B]

        if prob_A_culture > prob_B_culture:
            final_pred = cult_A
            final_prob = prob_A_culture
        else:
            final_pred = cult_B
            final_prob = prob_B_culture
        
        final_predictions.append({
            'Index': index,
            f'Prob_{cult_A}': prob_A_culture,
            f'Prob_{cult_B}': prob_B_culture,
            
            'Final_Prediction': final_pred,
            'Final_Confidence': final_prob
        })

    return pd.DataFrame(final_predictions).set_index('Index')

def write_summary(df: pd.DataFrame, output_path: str, filename: str):
        global_percent = df['result'].mean() * 100

        culture_summary = (
            df.groupby('culture')['result']
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        culture_summary_text = ""
        for culture, percent in culture_summary.items():
            culture_summary_text += f"Culture {culture}: {percent:.2f}%\n"

        summary_content = f"""
--- Prediction Summary Report ---

Overall Classification Success (Result == 1)

Total Instances: {len(df)}

Overall Success Percentage: {global_percent:.2f}%

---------------------------------------------------

## Success Rate by Culture Group

{culture_summary_text}
---------------------------------------------------
        """

        with open(f"{output_path}/{filename}", "w") as f:
            f.write(summary_content)

        print(f"Summary generated and saved to {output_path}/{filename}")

def main(args):
    parser = argparse.ArgumentParser(description="Predicting using Hexcrop models.")

    parser.add_argument("--input", nargs=1, type=str, help='The path to the CSV with features to predict')

    parser.add_argument("--unique", type=str, nargs=2, help="If you want to predict using one specific model, instead of cascading prediction")
    
    parser.add_argument("--output", nargs=1, type=str, help='The path to save the results.')

    parser.add_argument("--models", nargs=1, type=str, help='The path to load the models from.')

    args = parser.parse_args()

    # ==================== Parsing arguments ========================
    INPUT_PATH = "."
    INPUT_FILE = 'validation_dataset.csv'
    # INPUT_FILE = "full_feature_dataset.csv"
    if args.input:
        INPUT_PATH = args.input[0]

    UNIQUE_CULTURES = []
    RUN_SINGLE_PREDICTION = False
    if args.unique:
        lower = [un.lower() for un in args.unique]
        UNIQUE_CULTURES = [x.lower() for x in sorted(lower)]
        culture1, culture2 = UNIQUE_CULTURES
        RUN_SINGLE_PREDICTION = True

    OUTPUT_PATH = "."
    if args.output:
        OUTPUT_PATH=args.output[0]
        os.makedirs(OUTPUT_PATH, exist_ok=True)

    MODELS_PATH = "models"
    if args.models:
        MODELS_PATH = args.models[0]
    # ===============================================================

    try:
        data_to_predict_df = pd.read_csv(f"{INPUT_PATH}/{INPUT_FILE}")
    except FileNotFoundError:
        logger.error(f"Error: Data file not found at {INPUT_FILE}")
        return pd.DataFrame()

    if 'wheat' in list(data_to_predict_df['culture'].unique()):
        data_to_predict_df = data_to_predict_df[data_to_predict_df['culture'] != 'wheat']

    CULTURES_LIST = ["bean", "maize", "rice", "soybean"]#list(data_to_predict_df['culture'].unique())

    PAIRWISE_MODEL_MAP = create_pairwise_model_map(CULTURES_LIST, MODELS_PATH)

    if RUN_SINGLE_PREDICTION:
        data_to_predict_df = data_to_predict_df[data_to_predict_df['culture'].isin(UNIQUE_CULTURES)]
        cultures_column = data_to_predict_df['culture']
        filename_column = data_to_predict_df['filename']
        data_to_predict_df.drop(columns=['filename', 'year', 'season', 'culture'], inplace=True)

        GENERAL_MODEL_FEATURES = list(data_to_predict_df.columns)

        final_results_df = run_binary_prediction(data_to_predict_df,
            MODELS_PATH,
            GENERAL_MODEL_FEATURES,
            UNIQUE_CULTURES,
            PAIRWISE_MODEL_MAP
        )
        final_results_df['culture'] = cultures_column
        final_results_df['filename'] = filename_column
        final_results_df['result'] = (final_results_df['Final_Prediction'] == final_results_df['culture']).astype(int)
        final_results_df.to_csv(f"{OUTPUT_PATH}/results_{culture1}_{culture2}.csv")
        write_summary(final_results_df, OUTPUT_PATH, f"{culture1}_{culture2}_summary.txt")
    else:
        cultures_column = data_to_predict_df['culture']
        filename_column = data_to_predict_df['filename']
        data_to_predict_df.drop(columns=['filename', 'year', 'season', 'culture'], inplace=True)

        GENERAL_MODEL_FEATURES = list(data_to_predict_df.columns)
        final_results_df = run_cascading_prediction(
            data_to_predict_df,
            MODELS_PATH,
            GENERAL_MODEL_FEATURES,
            CULTURES_LIST,
            PAIRWISE_MODEL_MAP
        )
        final_results_df['culture'] = cultures_column
        final_results_df['filename'] = filename_column
        final_results_df['result'] = (final_results_df['Final_Prediction'] == final_results_df['culture']).astype(int)
        final_results_df.to_csv(f"{OUTPUT_PATH}/results_cascading.csv")
        write_summary(final_results_df, OUTPUT_PATH, "cascading_summary.txt")

if __name__ == "__main__":
    main(sys.argv)