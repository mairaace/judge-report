import pandas as pd
from pathlib import Path
from typing import List, Dict
from config import EVALUATED_MODELS, DATA_BASE_PATH, OUTPUT_PATH


def find_judge_runs(base_path: str, search_models: List[str]) -> Dict[str, List[str]]:
    """Find all judged.csv files for each judge."""
    judge_runs = {}
    base_path = Path(base_path)
    
    for judge_folder in base_path.iterdir():
        if not judge_folder.is_dir():
            continue
        
        judge_name = judge_folder.name
        
        for model_folder in judge_folder.iterdir():
            if not model_folder.is_dir():
                continue
            
            model_name = model_folder.name
            if not any(search_model in model_name for search_model in search_models):
                continue
            
            if judge_name not in judge_runs:
                judge_runs[judge_name] = []
            
            for rep_folder in model_folder.glob("temp_*"):
                judged_file = rep_folder / "judged.csv"
                if judged_file.exists():
                    judge_runs[judge_name].append(str(judged_file))
    
    return judge_runs


def _binarize_hard_truth(score: float) -> int:
    """Convert score to binary: 0 if ≤ 0.5, 1 if > 0.5"""  
    if score <= 0.5:
        return 0
    else:        
        return 1
    

def process_judge_data(file_paths: List[str]) -> pd.DataFrame:
    """Process multiple files from the same judge."""
    all_data = [pd.read_csv(file_path) for file_path in file_paths]
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Aggregation by question
    agg_dict = {
        'ground_truth': 'first',
        'answer': 'first',
        'judge_output_raw': 'first',
        'judge_model': 'first',
        'score__Veracidad': 'mean',
    }
    
    if 'score__Hard Truth' in combined_df.columns:
        agg_dict['score__Hard Truth'] = 'mean'
    
    result = combined_df.groupby(['index', 'question'], as_index=False).agg(agg_dict)
    
    # Binarize score_hard_truth
    if 'score__Hard Truth' in result.columns:
        result['score_hard_truth'] = result['score__Hard Truth'].apply(_binarize_hard_truth)
        result = result.drop(columns=['score__Hard Truth'])
    
    return result


def load_all_judges_data(base_path: str = None, models: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Load processed data for each judge."""
    if base_path is None:
        base_path = str(DATA_BASE_PATH)
    if models is None:
        models = EVALUATED_MODELS
    
    judge_runs = find_judge_runs(base_path, models)
    
    results = {}
    for judge_model, file_paths in judge_runs.items():
        results[judge_model] = process_judge_data(file_paths)
    
    return results


def save_judge_results(results: Dict[str, pd.DataFrame], output_path: str = None) -> None:
    """Save one CSV file per judge."""
    if output_path is None:
        output_path = str(OUTPUT_PATH)
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for judge_model, df in results.items():
        safe_name = judge_model.replace('/', '_').replace(' ', '_')
        output_file = output_path / f"judge_{safe_name}_processed.csv"
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    results = load_all_judges_data()
    if results:
        save_judge_results(results)