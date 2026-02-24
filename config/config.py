from pathlib import Path

# Get the project root (judge-report folder)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# evaluated models
EVALUATED_MODELS = [
'allenai_Llama-3.1-Tulu-3-70B_20260219_111108',
'allenai_Olmo-3.1-32B-Instruct_20260219_122119',
'cpt-sft-steps-8640_20260219_153051',
'deepseek_deepseek-v3.2_20260217_101816',
'google_gemma-3-27b-it_20260219_115742',
'gpt-4.1_20260219_125711',
'microsoft_phi-4_20260219_115102',
'mistralai_Mixtral-8x7B-Instruct-v0.1_20260219_112337',
'sft-phase1-135000_20260219_115547'
                    ]

# base paths
DATA_BASE_PATH = PROJECT_ROOT / 'data' / 'judge_data'
JUDGES_OUTPUT_PATH = PROJECT_ROOT / 'data' / 'judge_data_processed'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'outputs'
PILOT_CSV_PATH = PROJECT_ROOT / 'data' / 'calibration_data' / 'piloto.csv'
BENCHMARK_CSV_PATH = PROJECT_ROOT / 'data' / 'calibration_data' / 'Benchmark LatamGPT - QA consolidado 23 febrero.csv'

# Country filter for sample extraction
SAMPLE_COUNTRY = 'Chile'

#output folder setup
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# total sample size
M_TOTAL = 100



