import pandas as pd
import os
from itertools import combinations
import logging, sys
from utils import extract_all_features
from tqdm import tqdm

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

def calculate_features(input_folder: str):
    logger.info(f"Calculating all features from all files in 'limited' folders")
    data_columns = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "EVI2", "B1/B11", "B1/B12", "B1/B2", "B1/B3", "B1/B4", "B1/B5", "B1/B6", "B1/B7", "B1/B8", "B1/B9", "B11/B12", "B11/B2", "B11/B3", "B11/B4", "B11/B5", "B11/B6", "B11/B7", "B11/B8", "B11/B9", "B12/B2", "B12/B3", "B12/B4", "B12/B5", "B12/B6", "B12/B7", "B12/B8", "B12/B9", "B2/B3", "B2/B4", "B2/B5", "B2/B6", "B2/B7", "B2/B8", "B2/B9", "B3/B4", "B3/B5", "B3/B6", "B3/B7", "B3/B8", "B3/B9", "B4/B5", "B4/B6", "B4/B7", "B4/B8", "B4/B9", "B5/B6", "B5/B7", "B5/B8", "B5/B9", "B6/B7", "B6/B8", "B6/B9", "B7/B8", "B7/B9", "B8/B9"]
    all_feature_rows = []

    for root, dirs, files in os.walk(input_folder):
        if "limited" not in root.lower().replace("\\", "/"):
            continue

        path_parts = root.lower().replace("\\", "/").split("/")
        
        try:
            descriptor_index = path_parts.index("limited")
            season = path_parts[descriptor_index - 1]
            year = path_parts[descriptor_index - 2]
            culture = path_parts[descriptor_index - 3]
        except (ValueError, IndexError):
            culture = "unknown"
            season = "unknown"
            year = 'unknown'

        for file in tqdm(files, desc="Processing CSV files"):
            if file.lower().endswith(".csv"):
                filepath = os.path.join(root, file)
                
                try:
                    df = pd.read_csv(filepath)
                except Exception as e:
                    print(f"Error reading file {filepath}: {e}")
                    continue
                
                length = len(df)

                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                file_features = {
                    "filename": file,
                    "culture": culture,
                    "year": year,
                    "season": season,
                    "cycle_length": length
                }
                
                for column in data_columns:
                    features = extract_all_features(df, column)
                    for attr_name, attr_value in features.items():
                        file_features[f"{column}_{attr_name}"] = attr_value
                
                all_feature_rows.append(file_features)

    features_df = pd.DataFrame(all_feature_rows)
    return features_df

def calculate_bx_ratios(filepath: str, rewrite: bool = True):
    try:
        df = pd.read_csv(filepath)
        print(f"Calculating ratios for: {filepath}")

        bx_cols = sorted([col for col in df.columns if col.startswith('B') and col[1:].isdigit()])
        
        if not bx_cols:
            logger.warning(f"Warning: No 'BX' columns found in {filepath}. Skipping.")
            return

        ratio_pairs = list(combinations(bx_cols, 2))

        for col1, col2 in ratio_pairs:
            new_col_name = f"{col1}/{col2}"
            
            epsilon = 1e-6 
            df[new_col_name] = df[col1] / (df[col2] + epsilon)
            
        
        if rewrite:
            df.to_csv(filepath, index=False)
            print(f"Successfully calculated {len(ratio_pairs)} ratios and overwritten: {filepath}")
        else:
            df.to_csv(filepath.replace(".csv", "_indices.csv"), index=False)
            print(f"Successfully calculated {len(ratio_pairs)} ratios and written to: {filepath.replace('.csv', '_indices.csv')}")
          
    except Exception as e:
        print(f"Error processing {filepath}: {e}")


def add_indices_to_csvs(input_folder, rewrite: bool = True):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv') and file[0:2]=='id':
                filepath = os.path.join(root, file)
                calculate_bx_ratios(filepath)

    

def create_dataset(input_folder: str, output_path: str, rewrite: bool = True):
    if os.path.exists(input_folder):
        add_indices_to_csvs(input_folder, rewrite)
        features_df = calculate_features(input_folder)
    else:
        logger.error(f"Erro: nao achei {input_folder}")
        features_df = pd.DataFrame()

    if not features_df.empty:
        features_df.to_csv(output_path, index=False)

    return features_df

def split_dataset(df: pd.DataFrame, output_path: str, pct: float = 0.2):
    try:
        len_original = len(df)

        logger.info(f"Original dataset size: {len_original} rows.")

        validation_dataset = (
            df.groupby('culture', group_keys=False)
            .apply(lambda x: x.sample(frac=pct, random_state=37))
        )
        
        validation_indices = validation_dataset.index
        
        training_dataset = df.drop(validation_indices)

        len_validation = len(validation_dataset)
        len_training = len(training_dataset)
        
        logger.info(f"Validation dataset size: {len_validation} rows (approx {100 * len_validation/len_original}%).")
        logger.info(f"Training dataset size: {len_training} rows (approx {100 * len_training/len_original}%).")

        validation_dataset.to_csv(f"{output_path}/validation_dataset.csv", index=False)
        training_dataset.to_csv(f"{output_path}/training_dataset.csv", index=False)
        
        logger.info("\n✅ Success pai")

    except Exception as e:
        print(f"Erro :( {e}")
    pass

output_path = "."
full_features = create_dataset(output_path, f"{output_path}/full_dataset_features.csv", True)
split_dataset(full_features, output_path, 0.2)

