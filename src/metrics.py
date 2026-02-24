'''
Process judge evaluation data and calculate bias metrics.
'''
import sys
import pandas as pd
from pathlib import Path

# Fix path so config package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Tuple, Optional
from config.config import M_TOTAL, OUTPUT_PATH, PILOT_CSV_PATH, JUDGES_OUTPUT_PATH
from llm_judge_reporting import allocate_calibration_sample


def load_judge_pooled(judge_folder: Path) -> pd.DataFrame:
    """
    Read every model CSV inside a judge folder and concatenate them
    into a single DataFrame (keeps `evaluated_model` to distinguish rows).
    """
    frames = []
    for csv_file in sorted(judge_folder.glob("*.csv")):
        df = pd.read_csv(csv_file)
        frames.append(df)
        print(f"  Cargado: {csv_file.name}  ({len(df)} filas)")

    pooled = pd.concat(frames, ignore_index=True)
    print(f"  Total pooled: {len(pooled)} filas\n")
    return pooled



def calculate_p_metric(pooled_df: pd.DataFrame, judge_name: str) -> float:
    """Proportion of questions the judge marked as correct across ALL models."""
    correct = (pooled_df['score_hard_truth'] == 1).sum()
    total = len(pooled_df)
    p = correct / total if total > 0 else 0.0
    print(f"  [{judge_name}] p = {p:.4f}  ({correct}/{total})")
    return p



def calculate_q_metrics(
    pooled_df: pd.DataFrame,
    pilot_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    judge_name: str,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Calculate specificity (q0) and sensitivity (q1).

    Steps:
      1. Merge pilot with sample on `question` to recover `evaluated_model`.
      2. Merge that result with the judge's pooled data on
         (`question`, `evaluated_model`) so we compare the exact same answer.
      3. Compute q0 and q1.
    """
    #Normalise whitespace 
    pooled_df = pooled_df.copy()
    pilot_df = pilot_df.copy()
    sample_df = sample_df.copy()
    pooled_df['question'] = pooled_df['question'].str.strip()
    pilot_df['question'] = pilot_df['question'].str.strip()
    sample_df['question'] = sample_df['question'].str.strip()

     #(saber de qué modelo viene cada pregunta)
    pilot_with_model = pilot_df.merge(
        sample_df[['question', 'evaluated_model']],
        on='question',
        how='left',
    )
    
    #if we want to check the merging steps, we can print some info here whit the verbose parameter
    if verbose:
        n_pilot = len(pilot_df)
        n_bridge = pilot_with_model['evaluated_model'].notna().sum()
        print(f"\n  ┌─ PASO 1: Piloto → Sample (buscar evaluated_model)")
        print(f"  │  Preguntas en piloto : {n_pilot}")
        print(f"  │  Match con sample    : {n_bridge}/{n_pilot}")
        if n_bridge < n_pilot:
            missing = pilot_with_model[pilot_with_model['evaluated_model'].isna()]
            print(f"  │  ⚠ Sin match en sample:")
            for _, r in missing.iterrows():
                print(f"  │    - {r['question'][:80]}")
        print(f"  │")

        # Mostrar de qué modelo viene cada pregunta del piloto
        print(f"  │  {'#':>3}  {'Pregunta':<55} {'Modelo':>30}")
        print(f"  │  {'─'*3}  {'─'*55} {'─'*30}")
        for i, (_, r) in enumerate(pilot_with_model.iterrows(), 1):
            q = r['question'][:55]
            m = str(r['evaluated_model'])[:30] if pd.notna(r['evaluated_model']) else '???'
            print(f"  │  {i:>3}  {q:<55} {m:>30}")
        print(f"  └{'─'*92}\n")

    # Drop rows where we couldn't find the model
    pilot_with_model = pilot_with_model.dropna(subset=['evaluated_model'])

    # merge with judge's pooled data on (question, evaluated_model)
    merged = pilot_with_model.merge(
        pooled_df[['question', 'evaluated_model', 'score_hard_truth', 'answer']],
        on=['question', 'evaluated_model'],
        how='left',
    )
    
    #same, verbose is for checking the merging steps and the final comparison table
    if verbose:
        n_found = merged['score_hard_truth'].notna().sum()
        print(f"  ┌─ PASO 2: Comparación juez vs. humano ({judge_name})")
        print(f"  │  Pares (question, model) buscados : {len(merged)}")
        print(f"  │  Encontrados en datos del juez    : {n_found}/{len(merged)}")

        if n_found < len(merged):
            not_found = merged[merged['score_hard_truth'].isna()]
            print(f"  │  ⚠ No encontrados:")
            for _, r in not_found.iterrows():
                print(f"  │    - {r['question'][:60]} | modelo: {r['evaluated_model']}")
        print(f"  │")

        # Comparison table
        comp = merged.dropna(subset=['score_hard_truth']).copy()
        comp['judge_pred'] = (comp['score_hard_truth'] >= 0.5).astype(int)
        comp['human'] = comp['human_hard_truth'].astype(int)
        comp['ok'] = comp['judge_pred'] == comp['human']

        print(f"  │  {'#':>3}  {'Pregunta':<45} {'Modelo':<25} {'Juez':>5} {'Human':>6} {'OK?':>4}")
        print(f"  │  {'─'*3}  {'─'*45} {'─'*25} {'─'*5} {'─'*6} {'─'*4}")
        for i, (_, r) in enumerate(comp.iterrows(), 1):
            q = r['question'][:45]
            m = str(r['evaluated_model'])[:25]
            ok_str = '✓' if r['ok'] else '✗'
            print(f"  │  {i:>3}  {q:<45} {m:<25} {r['judge_pred']:>5} {r['human']:>6} {ok_str:>4}")

        total_ok = comp['ok'].sum()
        print(f"  │")
        print(f"  │  Coincidencias juez-humano: {total_ok}/{len(comp)} ({total_ok/len(comp)*100:.1f}%)")
        print(f"  └{'─'*92}\n")

    #Calculate q0, q1. the important metrics!!
    valid = merged.dropna(subset=['score_hard_truth'])
    judge_pred = (valid['score_hard_truth'] >= 0.5).astype(int)
    human = (valid['human_hard_truth'] == 1).astype(int)

    positives = human == 1
    negatives = human == 0

    q1 = (
        ((judge_pred == 1) & (human == 1)).sum() / positives.sum()
        if positives.sum() > 0 else 0.0
    )
    q0 = (
        ((judge_pred == 0) & (human == 0)).sum() / negatives.sum()
        if negatives.sum() > 0 else 0.0
    )

    print(f"  [{judge_name}] q0 (especificidad) = {q0:.4f},  q1 (sensibilidad) = {q1:.4f}")
    return q0, q1


# ──────────────────────────────────────────────
#  metrics per judge
# ──────────────────────────────────────────────

def calculate_all_metrics(
    pilot_csv_path: Optional[str] = None,
    outputs_folder: Optional[str] = None,
    judges_folder: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    this function get q0, q1 and p for each judge and pooled data and the pilot sample. and then 
    calculate the m0 and m1 for the calibration sample allocation.
    """
    if judges_folder is None:
        judges_folder = str(JUDGES_OUTPUT_PATH)
    if outputs_folder is None:
        outputs_folder = str(OUTPUT_PATH)
    if pilot_csv_path is None:
        pilot_csv_path = str(PILOT_CSV_PATH)

    judges_path = Path(judges_folder)
    outputs_path = Path(outputs_folder)
    pilot_df = pd.read_csv(pilot_csv_path)

    metrics_results: List[dict] = []

    for judge_dir in sorted(judges_path.iterdir()):
        if not judge_dir.is_dir():
            continue

        judge_name = judge_dir.name
        print(f"\n{'='*94}")
        print(f"  JUEZ: {judge_name}")
        print(f"{'='*94}\n")

        # Load & pool all model CSVs for this judge
        pooled_df = load_judge_pooled(judge_dir)

        # Load the sample CSV for this judge (to bridge pilot → evaluated_model)
        sample_pattern = f"sample_{judge_name}_*.csv"
        sample_files = list(outputs_path.glob(sample_pattern))
        if not sample_files:
            print(f"  ⚠ No se encontró sample para {judge_name} en {outputs_path}")
            print(f"    (buscando: {sample_pattern})")
            continue
        sample_df = pd.read_csv(sample_files[0])
        print(f"  Sample cargado: {sample_files[0].name}  ({len(sample_df)} filas)\n")

        # p metric (across all models)
        p_test = calculate_p_metric(pooled_df, judge_name)

        # q0, q1 (pilot comparison via sample bridge)
        q0_pilot, q1_pilot = calculate_q_metrics(
            pooled_df, pilot_df, sample_df, judge_name, verbose=verbose,
        )
        m_pilot_size = len(pilot_df)

        # Calibration allocation
        m0_target, m1_target = allocate_calibration_sample(
            m=M_TOTAL,
            p=p_test,
            q0_pilot=q0_pilot,
            q1_pilot=q1_pilot,
            m_pilot=m_pilot_size,
        )

        metrics_results.append({
            'judge_name': judge_name,
            'm_total': M_TOTAL,
            'p_test': round(p_test, 4),
            'q0_pilot': round(q0_pilot, 4),
            'q1_pilot': round(q1_pilot, 4),
            'm_pilot_size': m_pilot_size,
            'm0_target': m0_target,
            'm1_target': m1_target,
        })

    return pd.DataFrame(metrics_results)


# ──────────────────────────────────────────────
#   SAVE
# ──────────────────────────────────────────────

def save_metrics(metrics_df: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """Save metrics to CSV file."""
    if output_path is None:
        output_path = str(OUTPUT_PATH / 'judge_metrics.csv')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    print(f"\nMetrics saved to: {output_path}")


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    metrics_df = calculate_all_metrics(verbose=True)
    save_metrics(metrics_df)
    print("\n" + metrics_df.to_string(index=False))