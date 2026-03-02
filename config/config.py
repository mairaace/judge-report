from pathlib import Path

# Get the project root (judge-report folder)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# evaluated models
EVALUATED_MODELS = [
    "deepseek_deepseek-v3.2_20260217_101816",
    "google_gemma-3-27b-it_20260219_115742",
    "mistralai_mistral-large-2512_20260217_105922",
    "Qwen_Qwen2.5-72B-Instruct_20260219_113129",
                    ]

# base paths
DATA_BASE_PATH = PROJECT_ROOT / 'data' / 'few-judges'
JUDGES_OUTPUT_PATH = PROJECT_ROOT / 'data' / 'judge_data_processed'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'outputs'
PILOT_CSV_PATH = PROJECT_ROOT / 'data' / 'calibration_data' / 'piloto.csv'
EXTRA_QUESTIONS_PATH = PROJECT_ROOT / 'data' / 'outputs' / 'extra_questions.csv'
SAMPLE_CSV_PATH = PROJECT_ROOT / 'data' / 'outputs' / 'sample.csv'
BENCHMARK_CSV_PATH = PROJECT_ROOT / 'data' / 'calibration_data' / 'Benchmark LatamGPT - QA consolidado 23 febrero.csv'

# Country filter for sample extraction
SAMPLE_COUNTRY = 'Chile'

#output folder setup
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# total sample size
M_TOTAL = 100



