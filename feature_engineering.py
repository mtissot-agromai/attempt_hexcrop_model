import pandas as pd
from itertools import combinations
import logging, os, sys, argparse
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

def consolidate_feature_importances(importance_dir: str, importance_files: list) -> pd.DataFrame:
    all_importances = []

    for filename in importance_files:
        filepath = os.path.join(importance_dir, filename)
        try:
            model_name = filename.replace('_feature_importance.csv', '').replace('_', ' vs ')

            df_imp = pd.read_csv(filepath)
            df_imp = df_imp[['Feature', 'Importance']].copy()
            df_imp.rename(columns={'Importance': model_name}, inplace=True)

            all_importances.append(df_imp.set_index('Feature'))
            logger.info(f"Loaded importance for: {model_name}")

        except FileNotFoundError:
            logger.warning(f"Warning: Importance file not found: {filename}")
        except KeyError as e:
            logger.warning(f"Warning: Importance file {filename} is missing required column: {e}")

    if all_importances:
        combined_df = pd.concat(all_importances, axis=1).fillna(0)
        combined_df['Mean_Importance'] = combined_df.mean(axis=1)
        combined_df = combined_df.sort_values(by='Mean_Importance', ascending=False)
        return combined_df
    else:
        logger.error("No importance files were successfully loaded.")
        return pd.DataFrame()

def analyze_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'culture' not in df.columns:
        return pd.DataFrame()
    
    features = df.drop(columns=['culture'], errors='ignore').columns

    stats_df = df.groupby('culture')[features].agg(['mean', 'std', 'min', 'max', 'median'])

    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]

    logger.info("Statistics calculation complete.")
    return stats_df.T

def analyze_feature_correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    if df.empty or 'culture' not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    features_only_df = df.drop(columns=['culture'], errors='ignore')
    feature_corr_matrix = features_only_df.corr()
    logger.info("Feature-Feature correlation matrix computedd")

    label_dummies = pd.get_dummies(df['culture'], prefix='Label')
    corr_df = pd.concat([features_only_df, label_dummies], axis=1).corr()

    feature_label_corr = corr_df.loc[features_only_df.columns, label_dummies.columns]

    logger.info("Feature-Label correlation computed")
    return feature_corr_matrix, feature_label_corr

def main(args):
    parser = argparse.ArgumentParser(description="Performing feature analysis for Hexcrop models.")

    parser.add_argument("--input", nargs=1, type=str, help='The path to input folder.', required=True)

    parser.add_argument("--models", nargs=1, type=str, help='The path to models folder.', required=True)

    parser.add_argument("--output", nargs=1, type=str, help='The path to models folder.')

    args = parser.parse_args()

    # ========== Parsing arguments ==========
    INPUT_PATH = ''
    if args.input:
        INPUT_PATH = args.input[0]

    MODELS_DIR = ''
    if args.models:
        MODELS_DIR = args.models[0]

    OUTPUT_PATH = f'{MODELS_DIR}/features'
    if args.output:
        OUTPUT_PATH = args.output[0]

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # ======================================

    training_df = pd.read_csv(f"{INPUT_PATH}/training_dataset.csv")

    training_df.drop(columns=["filename", "season", "year"], inplace=True)

    IMPORTANCE_DIR = MODELS_DIR
    IMPORTANCE_FILES = os.listdir(IMPORTANCE_DIR)
    IMPORTANCE_FILES = [file for file in IMPORTANCE_FILES if file.endswith("_feature_importance.csv")]

    if not training_df.empty:
        combined_importances = consolidate_feature_importances(IMPORTANCE_DIR, IMPORTANCE_FILES)
        if not combined_importances.empty:
            combined_importances.to_csv(f"{OUTPUT_PATH}/consolidated_feature_importances.csv")

        feature_stats_by_label = analyze_feature_statistics(training_df)
        if not feature_stats_by_label.empty:
            feature_stats_by_label.to_csv(f"{OUTPUT_PATH}/feature_statistics_by_label.csv")

        feature_feature_corr, feature_label_corr = analyze_feature_correlations(training_df)

        if not feature_feature_corr.empty:
            feature_feature_corr.to_csv(f"{OUTPUT_PATH}/feature_feature_correlation_matrix.csv")

        if not feature_label_corr.empty:

            feature_label_corr['Max_Abs_Correlation'] = feature_label_corr.abs().max(axis=1)

            feature_label_corr.to_csv(f"{OUTPUT_PATH}/feature_culture_correlation.csv")

if __name__ == "__main__":
    main(sys.argv)