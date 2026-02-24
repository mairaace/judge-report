import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

# Fix path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import (
    EVALUATED_MODELS, DATA_BASE_PATH, OUTPUT_PATH,
    JUDGES_OUTPUT_PATH, M_TOTAL, BENCHMARK_CSV_PATH, SAMPLE_COUNTRY,
)


def find_judge_runs(base_path: str, search_models: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Find all judged.csv files organized by judge and evaluated model.
    Returns: {judge_name: {evaluated_model: [file_paths]}}
    """
    judge_runs = {}
    base_path = Path(base_path)
    print(f"Buscando en: {base_path}")
    print(f"Modelos a buscar: {search_models}")

    for judge_folder in base_path.iterdir():
        if not judge_folder.is_dir():
            continue

        judge_name = judge_folder.name
        print(f"\nJudge encontrado: {judge_name}")

        for model_folder in judge_folder.iterdir():
            if not model_folder.is_dir():
                continue

            model_name = model_folder.name
            if not any(search_model in model_name for search_model in search_models):
                print(f"  Modelo descartado: {model_name}")
                continue

            print(f"  Modelo válido: {model_name}")
            if judge_name not in judge_runs:
                judge_runs[judge_name] = {}
            if model_name not in judge_runs[judge_name]:
                judge_runs[judge_name][model_name] = []

            for rep_folder in sorted(model_folder.glob("temp_*")):
                judged_file = rep_folder / "judged.csv"
                if judged_file.exists():
                    print(f"    Archivo encontrado: {judged_file}")
                    judge_runs[judge_name][model_name].append(str(judged_file))
                else:
                    print(f"    No existe: {judged_file}")

    print(f"\nTotal judges encontrados: {len(judge_runs)}")
    return judge_runs


def _binarize_hard_truth(score: float) -> int:
    """Convert score to binary: 0 if ≤ 0.5, 1 if > 0.5"""
    if score <= 0.5:
        return 0
    else:
        return 1


def process_model_data(file_paths: List[str]) -> pd.DataFrame:
    """
    Process multiple run files for a single evaluated model under a single judge.
    Averages scores across runs and keeps the first run's answer.
    """
    all_data = [pd.read_csv(file_path) for file_path in file_paths]
    combined_df = pd.concat(all_data, ignore_index=True)

    # Aggregation by question (across runs)
    agg_dict = {
        'ground_truth': 'first',
        'answer': 'first',
        'judge_output_raw': 'first',
        'judge_model': 'first',
        'evaluated_model': 'first',
        'score__Veracidad': 'mean',
        'score__Hard Truth': 'mean',
    }

    result = combined_df.groupby(['index', 'question'], as_index=False).agg(agg_dict)

    # Binarize score_hard_truth
    result['score_hard_truth'] = result['score__Hard Truth'].apply(_binarize_hard_truth)
    result = result.drop(columns=['score__Hard Truth'])

    # Final column order
    result = result[['index', 'question', 'ground_truth', 'answer',
                      'judge_output_raw', 'judge_model', 'score__Veracidad',
                      'score_hard_truth', 'evaluated_model']]

    return result


def load_all_judges_data(
    base_path: str = None,
    models: List[str] = None
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load processed data for each judge, separated by evaluated model.
    Returns: {judge_name: {evaluated_model: DataFrame}}
    """
    if base_path is None:
        base_path = str(DATA_BASE_PATH)
    if models is None:
        models = EVALUATED_MODELS

    judge_runs = find_judge_runs(base_path, models)

    results = {}
    for judge_name, models_dict in judge_runs.items():
        results[judge_name] = {}
        for model_name, file_paths in models_dict.items():
            print(f"Procesando {judge_name} → {model_name} ({len(file_paths)} runs)")
            results[judge_name][model_name] = process_model_data(file_paths)

    return results


def save_judge_results(
    results: Dict[str, Dict[str, pd.DataFrame]],
    output_path: str = None
) -> None:
    """
    Save results as:  judge_data_processed/{judge_name}/{evaluated_model}.csv
    One CSV per evaluated model, inside a folder per judge.
    """
    if output_path is None:
        output_path = str(JUDGES_OUTPUT_PATH)

    output_path = Path(output_path)

    for judge_name, models_dict in results.items():
        safe_judge = judge_name.replace('/', '_').replace(' ', '_')
        judge_dir = output_path / safe_judge
        judge_dir.mkdir(parents=True, exist_ok=True)

        for model_name, df in models_dict.items():
            safe_model = model_name.replace('/', '_').replace(' ', '_')
            output_file = judge_dir / f"{safe_model}.csv"
            df.to_csv(output_file, index=False)
            print(f"Guardado: {output_file}  ({len(df)} filas)")


def extract_sample_from_judge(
    models_dict: Dict[str, pd.DataFrame],
    judge_name: str,
    sample_size: int = M_TOTAL,
    output_path: str = None,
    benchmark_csv_path: str = None,
    country: str = None,
) -> pd.DataFrame:
    """
    For a given judge, pool question/answer pairs from ALL evaluated models,
    filter to keep only questions from `country` (using the benchmark CSV to
    identify origin), remove duplicate questions (keep one random model per
    question), and take a random sample of `sample_size` rows.

    Output columns: question, ground_truth, answer, evaluated_model, País.
    """
    if benchmark_csv_path is None:
        benchmark_csv_path = str(BENCHMARK_CSV_PATH)
    if country is None:
        country = SAMPLE_COUNTRY

    # ── Load benchmark to get question → country mapping ──
    bench_df = pd.read_csv(benchmark_csv_path)
    bench_df['Pregunta'] = bench_df['Pregunta'].str.strip()
    country_questions = bench_df[bench_df['País'] == country][['Pregunta', 'País']].drop_duplicates(subset='Pregunta')
    print(f"Preguntas de {country} en benchmark: {len(country_questions)}")

    # ── Pool all models' data together ──
    all_rows = []
    for model_name, df in models_dict.items():
        subset = df[['question', 'ground_truth', 'answer', 'evaluated_model']].copy()
        all_rows.append(subset)

    pooled = pd.concat(all_rows, ignore_index=True)
    pooled['question'] = pooled['question'].str.strip()
    print(f"Total filas pooled (todos los modelos): {len(pooled)}")

    # ── Merge with benchmark to add country and filter ──
    pooled = pooled.merge(
        country_questions,
        left_on='question',
        right_on='Pregunta',
        how='inner',
    )
    pooled = pooled.drop(columns=['Pregunta'])
    print(f"Filas después de filtrar por {country}: {len(pooled)}")
    print(f"Preguntas únicas de {country}: {pooled['question'].nunique()}")

    # ── Remove duplicate questions: keep one random (question, model) pair ──
    pooled = pooled.sample(frac=1, random_state=42).reset_index(drop=True)
    pooled_unique = pooled.drop_duplicates(subset='question', keep='first')
    print(f"Filas tras eliminar preguntas duplicadas: {len(pooled_unique)}")

    # ── Random sample (without replacement) ──
    sample_df = pooled_unique.sample(
        n=min(sample_size, len(pooled_unique)),
        random_state=42,
    ).reset_index(drop=True)

    print(f"Muestra final: {len(sample_df)} filas")

    # ── Save ──
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        safe_name = judge_name.replace('/', '_').replace(' ', '_')
        output_file = output_path / f"sample_{safe_name}_{sample_size}.csv"
        sample_df.to_csv(output_file, index=False)
        print(f"Muestra guardada: {output_file}")

    return sample_df


if __name__ == "__main__":
    # Load all judges data (organized by judge → evaluated model)
    results = load_all_judges_data()

    if results:
        # Save processed results: one folder per judge, one CSV per model
        save_judge_results(results)

        # Extract random 100-question sample per judge 
        for judge_name, models_dict in results.items():
            extract_sample_from_judge(
                models_dict=models_dict,
                judge_name=judge_name,
                sample_size=M_TOTAL,
                output_path=str(OUTPUT_PATH)
            )