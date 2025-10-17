import pandas as pd
from itertools import combinations
import logging, os, sys, argparse
from utils import extract_all_features
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Dict, List, Tuple

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

# def interpolate_to_fixed_length(
#     df: pd.DataFrame, 
#     column: str, 
#     date_column: str, 
#     fixed_length: int = 30
# ) -> np.ndarray:
#     """
#     Resamples a single field's time series (NDVI, B8, etc.) to a fixed number
#     of equally spaced observation points using linear interpolation.
    
#     This standardizes the time axis so curves can be averaged.
    
#     Args:
#         df: DataFrame for a single field.
#         column: The spectral band or index column to process (e.g., 'NDVI', 'B8').
#         date_column: The name of the date column.
#         fixed_length: The target number of equally spaced points (N) in the prototype.
        
#     Returns:
#         A NumPy array of the resampled curve of length `fixed_length`, or 
#         an array of NaNs if insufficient data exists.
#     """
    
#     df_clean = df.dropna(subset=[column, date_column]).sort_values(date_column)
    
#     if len(df_clean) < 2:
#         # Cannot interpolate with less than 2 points
#         return np.full(fixed_length, np.nan)

#     # 1. Standardize Time Axis (t): Days Since Sowing (DSS)
#     df_clean[date_column] = pd.to_datetime(df_clean[date_column])
#     sowing_date = df_clean[date_column].min()
    
#     # x-axis: Time (in days)
#     time_points = (df_clean[date_column] - sowing_date).dt.days.values
    
#     # y-axis: Value (spectral band/index)
#     values = df_clean[column].values
    
#     # 2. Define the Target Fixed Time Grid
#     # Calculate the total length of the observed season for this field
#     max_time = time_points[-1] 
    
#     # Create the target fixed time grid (t_new) from 0 to max_time
#     # This creates 'fixed_length' points spanning the observed season
#     t_new = np.linspace(0, max_time, fixed_length)
    
#     # 3. Interpolate
#     # Use linear interpolation (cubic or spline could be used for smoother results)
#     interpolation_function = interp1d(time_points, values, kind='linear', fill_value='extrapolate')
#     resampled_curve = interpolation_function(t_new)
    
#     return resampled_curve

# def generate_crop_prototypes(
#     all_fields_data: List[Tuple[pd.DataFrame, str]],
#     spectral_columns: List[str],
#     date_column: str = 'date',
#     prototype_length: int = 30
# ) -> Dict[str, Dict[str, List[float]]]:
#     """
#     Performs the first pass over all field data to generate the mean prototype 
#     curve for every crop and every spectral column.
    
#     Args:
#         all_fields_data: A list of tuples, where each tuple is 
#                          (DataFrame for a single field, Crop Label string).
#         spectral_columns: A list of all band, ratio, and index columns 
#                           (e.g., ['B4', 'B8', 'NDVI', 'B8/B4']).
#         date_column: The name of the date column.
#         prototype_length: The standardized length (N) of the resulting prototype curve.
        
#     Returns:
#         A nested dictionary: {Column_Name: {Crop_Label: [Prototype_Curve_Values]}}
#     """
    
#     # Structure to hold all resampled curves before averaging
#     # {Column: {Crop: [list_of_Numpy_Arrays_of_size_N]}}
#     collected_curves: Dict[str, Dict[str, List[np.ndarray]]] = {col: {} for col in spectral_columns}
    
#     print(f"Starting prototype generation with fixed length N={prototype_length}...")

#     # PASS 1: Resample all individual field curves
#     for df_field, crop_label in all_fields_data:
        
#         # Ensure the crop label exists in the collection structure
#         for col in spectral_columns:
#             if crop_label not in collected_curves[col]:
#                 collected_curves[col][crop_label] = []

#         for col in spectral_columns:
#             # Resample the irregular time series to the fixed N points
#             resampled = interpolate_to_fixed_length(df_field, col, date_column, prototype_length)
            
#             # Only store valid curves (not full of NaNs)
#             if not np.all(np.isnan(resampled)):
#                 collected_curves[col][crop_label].append(resampled)

#     # PASS 2: Aggregate (Calculate the mean prototype curve)
#     master_prototypes: Dict[str, Dict[str, List[float]]] = {}
    
#     for col, crop_data in collected_curves.items():
#         master_prototypes[col] = {}
#         print(f"  Aggregating prototypes for column: {col}")
        
#         for crop_label, curves_list in crop_data.items():
#             if not curves_list:
#                 print(f"    Warning: No valid data for {col} - {crop_label}.")
#                 continue

#             # Stack all arrays for this crop/column and calculate the mean along the time axis (axis=0)
#             stacked_arrays = np.stack(curves_list, axis=0)
            
#             # The prototype is the mean (average) of all individual resampled curves
#             mean_prototype = np.nanmean(stacked_arrays, axis=0)
            
#             # Store the resulting prototype (convert back to list for final format)
#             master_prototypes[col][crop_label] = mean_prototype.tolist()
#             print(f"    Generated prototype for {crop_label} from {len(curves_list)} fields.")

#     return master_prototypes

# def create_master_prototypes(input_folder: str):
#     data_columns = ['B1', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
#        'B8A', 'B9', 'EVI2', 'B1/B11', 'B1/B12', 'B1/B2', 'B1/B3', 'B1/B4',
#        'B1/B5', 'B1/B6', 'B1/B7', 'B1/B8', 'B1/B8A', 'B1/B9', 'B11/B12',
#        'B11/B2', 'B11/B3', 'B11/B4', 'B11/B5', 'B11/B6', 'B11/B7', 'B11/B8',
#        'B11/B8A', 'B11/B9', 'B12/B2', 'B12/B3', 'B12/B4', 'B12/B5', 'B12/B6',
#        'B12/B7', 'B12/B8', 'B12/B8A', 'B12/B9', 'B2/B3', 'B2/B4', 'B2/B5',
#        'B2/B6', 'B2/B7', 'B2/B8', 'B2/B8A', 'B2/B9', 'B3/B4', 'B3/B5', 'B3/B6',
#        'B3/B7', 'B3/B8', 'B3/B8A', 'B3/B9', 'B4/B5', 'B4/B6', 'B4/B7', 'B4/B8',
#        'B4/B8A', 'B4/B9', 'B5/B6', 'B5/B7', 'B5/B8', 'B5/B8A', 'B5/B9',
#        'B6/B7', 'B6/B8', 'B6/B8A', 'B6/B9', 'B7/B8', 'B7/B8A', 'B7/B9',
#        'B8/B8A', 'B8/B9', 'B8A/B9', 'NDVI', 'NDWI','NDMI']
#     ALL_FIELDS_DATA = []
#     for root, dirs, files in os.walk(input_folder):
#         if "limited" not in root.lower().replace("\\", "/"):
#             continue

#         path_parts = root.lower().replace("\\", "/").split("/")
        
#         try:
#             descriptor_index = path_parts.index("limited")
#             season = path_parts[descriptor_index - 1]
#             year = path_parts[descriptor_index - 2]
#             culture = path_parts[descriptor_index - 3]
#         except (ValueError, IndexError):
#             culture = "unknown"
#             season = "unknown"
#             year = 'unknown'

#         for file in tqdm(files, desc="Processing CSV files"):
#             if file.lower().endswith(".csv"):
#                 filepath = os.path.join(root, file)
#                 try:
#                     df = pd.read_csv(filepath)
#                 except Exception as e:
#                     df = pd.DataFrame()
#                     logger.error(f"Erro na leitura do csv: {e}")
#                 if not df.empty and culture != 'unknown':
#                     ALL_FIELDS_DATA.append((df, culture))

    
#     return generate_crop_prototypes(ALL_FIELDS_DATA, data_columns, prototype_length=30)

def calculate_features(input_folder: str, master_prototypes = None):
    logger.info(f"Calculating all features from all files in 'limited' folders")
    data_columns = ['B1', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'EVI2', 'B1/B11', 'B1/B12', 'B1/B2', 'B1/B3', 'B1/B4',
       'B1/B5', 'B1/B6', 'B1/B7', 'B1/B8', 'B1/B8A', 'B1/B9', 'B11/B12',
       'B11/B2', 'B11/B3', 'B11/B4', 'B11/B5', 'B11/B6', 'B11/B7', 'B11/B8',
       'B11/B8A', 'B11/B9', 'B12/B2', 'B12/B3', 'B12/B4', 'B12/B5', 'B12/B6',
       'B12/B7', 'B12/B8', 'B12/B8A', 'B12/B9', 'B2/B3', 'B2/B4', 'B2/B5',
       'B2/B6', 'B2/B7', 'B2/B8', 'B2/B8A', 'B2/B9', 'B3/B4', 'B3/B5', 'B3/B6',
       'B3/B7', 'B3/B8', 'B3/B8A', 'B3/B9', 'B4/B5', 'B4/B6', 'B4/B7', 'B4/B8',
       'B4/B8A', 'B4/B9', 'B5/B6', 'B5/B7', 'B5/B8', 'B5/B8A', 'B5/B9',
       'B6/B7', 'B6/B8', 'B6/B8A', 'B6/B9', 'B7/B8', 'B7/B8A', 'B7/B9',
       'B8/B8A', 'B8/B9', 'B8A/B9', 'NDVI', 'NDWI','NDMI']

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
                    if master_prototypes is None:
                        features = extract_all_features(df, column)
                    else:
                        features = extract_all_features(df, column, master_prototypes)
                    for attr_name, attr_value in features.items():
                        file_features[f"{column}_{attr_name}"] = attr_value
                
                all_feature_rows.append(file_features)

    features_df = pd.DataFrame(all_feature_rows)
    return features_df

def calculate_bx_ratios(filepath: str, rewrite: bool = True):
    try:
        df = pd.read_csv(filepath)
        print(f"Calculating ratios for: {filepath}")

        bx_cols = sorted([col for col in df.columns if (col.startswith('B') and col[1:].isdigit()) or col=='B8A'])
        
        if not bx_cols:
            logger.warning(f"Warning: No 'BX' columns found in {filepath}. Skipping.")
            return

        ratio_pairs = list(combinations(bx_cols, 2))

        for col1, col2 in ratio_pairs:
            new_col_name = f"{col1}/{col2}"
            
            epsilon = 1e-6 
            df[new_col_name] = df[col1] / (df[col2] + epsilon)

        df['NDVI'] = (df['B8'] - df['B4'])/(df['B8'] + df['B4']) # 1.6 * 10000
        df['NDWI'] = (df['B8'] - df['B11'])/(df['B8'] + df['B11'])
        df['NDMI'] = (df['B8'] - df['B12'])/(df['B8'] + df['B12']) 
        
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
            if file.endswith('.csv'):
                filepath = os.path.join(root, file)
                calculate_bx_ratios(filepath, rewrite)

def create_dataset(input_folder: str, output_path: str, rewrite: bool = True):
    if os.path.exists(input_folder):
        add_indices_to_csvs(input_folder, rewrite)
        # master_prototypes = create_master_prototypes(input_folder)

        # features_df = calculate_features(input_folder, master_prototypes)
        features_df = calculate_features(input_folder)
    else:
        logger.error(f"Erro: nao achei {input_folder}")
        features_df = pd.DataFrame()

    if not features_df.empty:
        features_df.to_csv(output_path, index=False)

    features_df.to_csv(output_path, index=False)

    return features_df

def split_dataset(df: pd.DataFrame, output_path: str, pct: float, rseed: int = 123):
    try:
        len_original = len(df)

        logger.info(f"Original dataset size: {len_original} rows.")

        validation_dataset = (
            df.groupby('culture', group_keys=False)
            .apply(lambda x: x.sample(frac=pct, random_state=rseed))
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

def main(args):
    parser = argparse.ArgumentParser(description="Creating dataset for Hexcrop models.")

    parser.add_argument("--nosplit", 
                        action='store_true',
                        dest='SPLIT_DATASET',
                        default=True,
                        help='Do not split the dataset into training and validation datasets')
    
    parser.add_argument("--split", nargs=1, type=float, help='The proportion for the velidation dataset')
    
    parser.add_argument("--output", nargs=1, type=str, help='The path to save the dataset(s).')

    parser.add_argument("--input", nargs=1, type=str, help='The path to input folder.')

    parser.add_argument("--rseed", nargs=1, type=int, help='The path to input folder.')

    args = parser.parse_args()


    # ========== Parsing arguments ==========
    SPLIT_DATASET = args.SPLIT_DATASET

    OUTPUT_PATH='.'
    if args.output:
        OUTPUT_PATH=args.output[0]
        os.makedirs(OUTPUT_PATH, exist_ok=True)

    rseed = None
    if args.rseed:
        rseed = args.rseed[0]

    INPUT_FOLDER = "culture_data"
    if args.input:
        INPUT_FOLDER = args.input[0]

    VALIDATION_PROPORTION = 0.2
    if args.split:
        VALIDATION_PROPORTION = args.split[0]
        if VALIDATION_PROPORTION < 0:
            print(f"This is a proportion. It should be between 0 and 1. {VALIDATION_PROPORTION} is less than 0. Default to 0.2")
            VALIDATION_PROPORTION = 0.2
        if VALIDATION_PROPORTION > 1:
            print(f"This is a proportion. It should be between 0 and 1. {VALIDATION_PROPORTION} is bigger than 1. Defaulting to 0.2")
            VALIDATION_PROPORTION = 0.2

    FILENAME = "full_feature_dataset"
    # ======================================

    full_features = create_dataset(INPUT_FOLDER, f"{OUTPUT_PATH}/{FILENAME}.csv", True)

    if SPLIT_DATASET:
        if rseed:
            split_dataset(full_features, OUTPUT_PATH, VALIDATION_PROPORTION, rseed)
        else:
            split_dataset(full_features, OUTPUT_PATH, VALIDATION_PROPORTION)

if __name__ == "__main__":
    main(sys.argv)