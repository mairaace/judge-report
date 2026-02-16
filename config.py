
from pathlib import Path

# Modelos evaluados
EVALUATED_MODELS = ['meta-llama_Llama-3.1-70B-Instruct_20260204_174748']

# Rutas base
PROJECT_ROOT = Path(__file__).parent  # judge-report/
DATA_BASE_PATH = PROJECT_ROOT / 'data'
OUTPUT_PATH = PROJECT_ROOT / 'outputs'

# Crear directorio de salida si no existe
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

M_TOTAL = 100  # presupuesto total de revisión humana
