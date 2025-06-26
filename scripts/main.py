import os
import pandas as pd
from scripts.inference import process_csv_file
from scripts.utils import get_filename_suffix, save_mcq_file
from functools import partial
import yaml
from scripts.models import (
    get_prediction_mistral,
    get_prediction_fanar,
    get_prediction_allam,
    get_prediction_fanar_validated,           
    get_prediction_fanar_validated_detailed   
)


# Mapping models to functions
MODEL_FUNCTIONS = {
    "mistral": partial(get_prediction_mistral, model_version="mistral-saba-24b"),
    "fanar_rag": partial(get_prediction_fanar, model_version="Islamic-RAG"),
    "fanar_validated": get_prediction_fanar_validated,           
    "fanar_validated_detailed": get_prediction_fanar_validated_detailed,  
    "allam_7b": partial(get_prediction_allam, model_version="ALLaM-AI/ALLaM-7B-Instruct-preview"),
}

def load_config(config_path: str):
    """
    Load YAML configuration file, enforce single model selection.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    input_dir = config.get("paths", {}).get("input_dir")
    output_dir = config.get("paths", {}).get("output_dir")

    # Select all models marked "Y"
    selected_models_all = [
        model for model, status in config.get("models", {}).items() if status.strip().upper() == "Y"
    ]

    if not selected_models_all:
        raise ValueError("❌ No models selected in configuration file.")

    # Take only the first model
    selected_model = selected_models_all[0]

    if len(selected_models_all) > 1:
        print(f"⚠ Multiple models selected ({selected_models_all}), using only the first: {selected_model}")

    return input_dir, output_dir, [selected_model]


def predict_from_directory(config_path="../config.yaml") -> pd.DataFrame:
    """
    Batch process all CSV files in input directory, run predictions, and save output.
    """
    all_predictions_df = pd.DataFrame()
    input_dir, output_dir, selected_models = load_config(config_path)
    model_suffix = "_".join(selected_models)
    print(f"✅ Model selected: {selected_models[0]}")
    if not selected_models:
        return all_predictions_df
    models_to_evaluate = {model: MODEL_FUNCTIONS[model] for model in selected_models if model in MODEL_FUNCTIONS}
    for file in os.listdir(input_dir):
        if file.endswith(".csv") and "_prediction" not in file:
            try:
                print(f"🚀 Processing file: {file}")
                input_path = os.path.join(input_dir, file)
                df = process_csv_file(input_path, models_to_evaluate)
                if df is None:
                    print(f"⚠️ Skipping file {file} due to invalid format or error.")
                    continue
                all_predictions_df = pd.concat([all_predictions_df, df], ignore_index=True)
                # Rename validated column as 'prediction' for output
                model_name = list(models_to_evaluate.keys())[0]
                if model_name + '_validated' in df.columns:
                    df = df.rename(columns={model_name + '_validated': 'prediction'})
                # Reorder columns to place 'prediction' after 'level' if present
                if 'level' in df.columns and 'prediction' in df.columns:
                    cols = list(df.columns)
                    cols.remove('prediction')
                    level_idx = cols.index('level')
                    cols = cols[:level_idx+1] + ['prediction'] + cols[level_idx+1:]
                    df = df[cols]
                # Only call get_filename_suffix if df is a DataFrame
                if isinstance(df, pd.DataFrame):
                    suffix = get_filename_suffix(df)
                    output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_{model_suffix}_subtask{suffix}_prediction.csv")
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    df.to_csv(output_file, index=False)
                    save_mcq_file(output_file)
            except Exception as e:
                print(f"❌ Error processing file {file}: {e}")
    return all_predictions_df
