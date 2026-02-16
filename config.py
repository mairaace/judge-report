
from pathlib import Path

# evaluated models
EVALUATED_MODELS = ['meta-llama_Llama-3.1-70B-Instruct_20260204_174748']

# base paths
PROJECT_ROOT = Path(__file__).parent  # judge-report/
DATA_BASE_PATH = PROJECT_ROOT / 'data' / 'judge_data'
JUDGES_OUTPUT_PATH = PROJECT_ROOT / 'data' / 'judge_data_processed'  # CSVs procesados de los jueces
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'outputs'  # Donde guardar resultados
PILOT_CSV_PATH = PROJECT_ROOT / 'data' / 'pilot.csv'

# output folder setup
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# total sample size
M_TOTAL = 100  

