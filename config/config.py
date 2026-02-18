from pathlib import Path

# Get the project root (judge-report folder)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# evaluated models
EVALUATED_MODELS = ['meta-llama_Llama-3.1-70B-Instruct_20260204_174748']

# base paths
DATA_BASE_PATH = PROJECT_ROOT / 'data' / 'judge_data'
JUDGES_OUTPUT_PATH = PROJECT_ROOT / 'data' / 'judge_data_processed'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'outputs'
PILOT_CSV_PATH = PROJECT_ROOT / 'data' / 'calibration_data' / 'pilot.csv'

# output folder setup
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# total sample size
M_TOTAL = 100