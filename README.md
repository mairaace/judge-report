# Judge Report

This repo estimates the accuracy of an LLM-based judge by comparing its evaluations against human ground truth. It follows a systematic three-stage workflow: preprocessing to extract a pilot sample, manual evaluation of that sample, and metric calculation to determine calibration sample sizes.

## Requirements 

this repo is based in this project:

```bash
pip install "git+https://github.com/UW-Madison-Lee-Lab/LLM-judge-reporting.git"
```


### Stage 1: Preprocessing

The preprocessing step extracts a representative pilot sample from your evaluated models.

**Input data needed:**
- Judge evaluation files (organized by judge name and model in `data/judge_data/`)
- A benchmark CSV file with question and answer pairs
- Configuration specifying which models were evaluated

**Run preprocessing:**
```bash
python src/preprocessing.py
```

**Output:**
- `data/outputs/sample_Qwen3-Next-80B-A3B-Instruct_100.csv` - 100 chilean cuestions to extract de pilot 30 questions.


**What it does:**
- Scans all judge evaluation results for the specified models
- Pools evaluations across all judges
- Extracts a stratified sample (default: filtered by country) for manual human evaluation

### Stage 2: Manual Evaluation

After preprocessing generates the pilot, a human annotators must manually evaluate each question in the pilot sample.

**Requirements:**
- Open `data/calibration_data/piloto.csv`
- For each question, add human ground truth labels (1 = correct, 0 = incorrect)
- Save the updated file with the column `human_hard_truth`

This manually-labeled pilot becomes your ground truth for evaluating judge accuracy.

### Stage 3: Metric Calculation

The metrics script compares judge predictions against the human-labeled pilot to quantify judge reliability.

**Run metrics:**
```bash
python src/metrics.py
```

**Input:**
- `data/calibration_data/piloto.csv` - The human-labeled pilot sample
- Judge evaluation files from `data/judge_data/` (pooled per judge)
- Sample files from `data/outputs/` (to link pilot questions to evaluated models)

**Output:**
- Console logging of comparative analysis
- `data/outputs/judge_metrics.csv` - Summary table with:
  - `q0` - Judge specificity (% correct on negative cases)
  - `q1` - Judge sensitivity (% correct on positive cases)
  - `p_test` - Overall proportion of positive labels in judge's pool
  - `m0_target`, `m1_target` - Recommended calibration sample sizes for false negatives and positives

**What it calculates:**
- Match rate between judge predictions and human ground truth
- Specificity and sensitivity metrics per judge
- Calibration sample allocation (how many negative and positive examples to label for direct validation)

### Stage 4: Final Estimation

*In progress.* 
## Configuration

Edit `config/config.py` to customize:
- `EVALUATED_MODELS` - List of model names to process
- `SAMPLE_COUNTRY` - Geographic filter for pilot extraction
- `M_TOTAL` - Calibration sample size
- Data paths: `DATA_BASE_PATH`, `JUDGES_OUTPUT_PATH`, `OUTPUT_PATH`

## File Structure

```
data/
  judge_data/              # Raw judge evaluation outputs (by judge and model)
  judge_data_processed/    # Pooled judge data per judge
  calibration_data/        # Pilot sample and benchmark files
  outputs/                 # Generated metrics and samples
src/
  preprocessing.py    # Extract pilot sample
  metrics.py          # Calculate judge metrics
config/
  config.py           # Configuration and paths
main.py              # Final estimation pipeline (in progress)
```




