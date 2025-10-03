import pandas as pd
import numpy as np
import onnxruntime as rt
from itertools import combinations
from typing import Dict, Tuple, List, Any

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

def get_onnx_input_name(session: rt.InferenceSession) -> str:
    return session.get_inputs()[0].name

def create_pairwise_model_map(cultures: List[str], models_path: str) -> Dict[Tuple[str, str], str]:
    model_map = {}
    
    for cult1, cult2 in combinations(sorted(cultures), 2):
        filename = f"{models_path}/{cult1}_{cult2}.onnx"
        
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
    new_data_path: str,
    general_model_path: str,
    full_feature_list: List[str],
    class_list: List[str],
    SAVE: bool = False
) -> pd.DataFrame:
    
    try:
        new_data = pd.read_csv(new_data_path, index_col=0)
    except FileNotFoundError:
        logger.error(f"Error: Data file not found at {new_data_path}")
        return pd.DataFrame()

    try:
        general_sess = rt.InferenceSession(f"{general_model_path}/general_model.onnx")
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
            pairwise_sess = rt.InferenceSession(model_file)
        except Exception:
            logger.warning(f"Warning: Could not load pairwise model {model_file}. Skipping sample.")
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

models_path = "./models"
output_path = "./result_multimodel.csv"
data_to_predict_path = "./validation_dataset.csv"

df = pd.read_csv(data_to_predict_path)

# " IF YOU DROPPED WHEAT FROM THE THING BEFORE, DROP HERE TOO"
if 'wheat' in list(df['culture'].unique()):
    df = df[df['culture'] != 'wheat']
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

CULTURES_LIST = list(df['culture'].unique())

df = df.drop(columns=["filename", "culture", "year", "season"])

GENERAL_MODEL_FEATURES = list(df.columns)
del df

PAIRWISE_MODEL_MAP = create_pairwise_model_map(CULTURES_LIST, models_path)

final_results_df = run_cascading_prediction(
    data_to_predict_path, 
    models_path, 
    GENERAL_MODEL_FEATURES, 
    CULTURES_LIST
)

final_results_df.to_csv(f"{output_path}")