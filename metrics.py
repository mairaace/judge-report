'''
Process judge evaluation data and calculate bias metrics.
'''
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from config import M_TOTAL, OUTPUT_PATH, PILOT_CSV_PATH, JUDGES_OUTPUT_PATH
from llm_judge_reporting import allocate_calibration_sample


def process_judge_data(outputs_folder: str = None) -> Dict[str, pd.DataFrame]:
    """Load all judge CSV files from outputs folder."""
    if outputs_folder is None:
        outputs_folder = str(JUDGES_OUTPUT_PATH)
    
    outputs_path = Path(outputs_folder)
    judge_data = {}
    
    for csv_file in outputs_path.glob("judge_*_processed.csv"):
        df = pd.read_csv(csv_file)
        judge_data[csv_file.stem] = df
    
    return judge_data


def calculate_p_metric(df: pd.DataFrame, judge_name: str) -> float:
    """Calculate proportion of questions judged as correct."""
    correct_count = (df['score_hard_truth'] == 1).sum()
    total_count = len(df)
    p_metric = correct_count / total_count
    
    print(f"{judge_name}: p={p_metric:.4f} ({correct_count}/{total_count})")
    return p_metric


def calculate_q_metrics(df: pd.DataFrame, pilot_df: pd.DataFrame, judge_name: str) -> Tuple[float, float]:
    """Calculate sensitivity (q1) and specificity (q0) from pilot data."""
    merged = df.merge(pilot_df[['question', 'human_hard_truth']], on='question', how='inner')
    
    judge_predictions = (merged['score_hard_truth'] >= 0.5).astype(int)
    human_truth = (merged['human_hard_truth'] == 1).astype(int)
    
    positives = human_truth == 1
    negatives = human_truth == 0
    
    q1_pilot = ((judge_predictions == 1) & (human_truth == 1)).sum() / positives.sum() if positives.sum() > 0 else 0.0
    q0_pilot = ((judge_predictions == 0) & (human_truth == 0)).sum() / negatives.sum() if negatives.sum() > 0 else 0.0
    
    print(f"{judge_name}: q0={q0_pilot:.4f}, q1={q1_pilot:.4f} (pilot size={len(merged)})")
    return q0_pilot, q1_pilot

def calculate_all_metrics(pilot_csv_path: Optional[str] = None, outputs_folder: Optional[str] = None) -> pd.DataFrame:
    """Calculate all metrics for each judge."""
    if outputs_folder is None:
        outputs_folder = str(JUDGES_OUTPUT_PATH)
    if pilot_csv_path is None:
        pilot_csv_path = str(PILOT_CSV_PATH)
    
    judge_data = process_judge_data(outputs_folder)
    pilot_df = pd.read_csv(pilot_csv_path)
    
    metrics_results = []
    
    for judge_name, judge_df in judge_data.items():
        evaluated_model = judge_df['judge_model'].iloc[0]
        p_test = calculate_p_metric(judge_df, judge_name)
        
        q0_pilot, q1_pilot = calculate_q_metrics(judge_df, pilot_df, judge_name)
        m_pilot_size = len(pilot_df)
        
        m0_target, m1_target = allocate_calibration_sample(
            m=M_TOTAL,
            p=p_test,
            q0_pilot=q0_pilot,
            q1_pilot=q1_pilot,
            m_pilot=m_pilot_size
        )
        
        metrics_results.append({
            'judge_name': judge_name,
            'evaluated_model': evaluated_model,
            'm_total': M_TOTAL,
            'p_test': p_test,
            'q0_pilot': q0_pilot,
            'q1_pilot': q1_pilot,
            'm_pilot_size': m_pilot_size,
            'm0_target': m0_target,
            'm1_target': m1_target,
        })
    
    return pd.DataFrame(metrics_results)


def save_metrics(metrics_df: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """Save metrics to CSV file."""
    if output_path is None:
        output_path = str(OUTPUT_PATH / 'judge_metrics.csv')
    
    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    metrics_df = calculate_all_metrics()
    save_metrics(metrics_df)
    print(metrics_df.to_string(index=False))