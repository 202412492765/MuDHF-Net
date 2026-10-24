# MuDHF-Net

MuDHF-Net: An Interpretable Deep Learning Framework with Representation Disentanglement and Hypergraph Modeling for Multi-Omics Cancer Prognosis Prediction

## Installation

Python 3.9 is supported. Install the dependencies in a dedicated environment:

```bash
python -m pip install -r requirements.txt
```

The original training environment used Python 3.9, PyTorch 2.3.1, and CUDA 12.1. Install the matching PyTorch CUDA build when reproducing GPU training. CPU execution is supported.

## Repository layout

```text
configs/           # Training configuration, hyperparameter search range, and selected                      parameters for various cancer types after experimentation
data/              # Prepared omics matrices and survival endpoints
data/masks/        # Methylation observation masks
splits/            # Fixed five-fold assignments
trained/           # MSPAE weights, four representations, CI/AUC fold checkpoints
src/models/        # MSPAE, HGNN, attention fusion, objectives
src/utils/         # Data loading, hypergraphs, metrics, deterministic configuration
src/scripts/       # Three execution entrypoints
results/           # Recorded five-fold metrics
```

## Data

Cohorts: BRCA (749), LUAD (433), STAD (343), LIHC (339), BLCA (390).

Each cohort directory contains `{cohort}_mrna.csv` (2000 features), `{cohort}_mirna.csv` (500 features), `{cohort}_meth.csv` (500 features), and `{cohort}_clinical.csv` (`time`, `event`). Rows are patients; the first column is `patient_id`. Survival time is in days; event is 1 for death and 0 for censoring. Input values are supplied in their prepared numerical scale. Methylation masks contain 1 for observed and 0 for unobserved reconstruction targets. Fold assignments use integers 0 through 4.

Representations are `shared.csv`, `private_mirna.csv`, `private_meth.csv`, and `private_mrna.csv`, each with 64 dimensions and the same patient order. Column order in input matrices must remain unchanged.

## Run

Run commands from the repository root. Choose `--task CI` or `--task AUC` separately.

### 1. Train MSPAE and downstream prediction

```bash
python -m src.scripts.run_pipeline --cohort BRCA --task CI --mode selected
python -m src.scripts.run_pipeline --cohort BRCA --task AUC --mode grid
```

### 2. Train prediction from saved representations

```bash
python -m src.scripts.run_from_representations --cohort BRCA --task CI --mode selected
python -m src.scripts.run_from_representations --cohort BRCA --task AUC --mode grid
```

### 3. Regenerate representations from MSPAE weights and train prediction

```bash
python -m src.scripts.run_from_checkpoint --cohort BRCA --task CI --mode selected
python -m src.scripts.run_from_checkpoint --cohort BRCA --task AUC --mode grid
```

`selected` retrains all five folds with the supplied selected parameters. `grid` searches 100 ranking/CVaR weight pairs over the same five folds. `evaluate` reloads the supplied downstream checkpoints:

```bash
python -m src.scripts.run_from_representations --cohort BRCA --task CI --mode evaluate
python -m src.scripts.run_from_checkpoint --cohort BRCA --task AUC --mode evaluate
```

Use `--device cpu` or `--device cuda:0` to select a device. `--output PATH` selects a new empty output directory. `--representations PATH` and `--checkpoint PATH` select alternative saved artifacts for their respective entrypoints. `--root PATH` selects the repository data root. `--config PATH` loads an alternative training configuration.

## Training protocol and outputs

MSPAE is fitted once on the complete cohort using omics inputs only. Its objective combines reconstruction, shared alignment , and shared/private orthogonality. Training stops after 60 epochs without meaningful objective improvement, following a minimum of 100 epochs; the default has no hard epoch cap. The minimum-objective checkpoint supplies four 64-dimensional representations.

Each representation has a separate cosine-neighbor hypergraph and HGNN branch. Four learned queries fuse the branches through four-head cross-attention. Hypergraphs are built separately for the training and validation patients of each fold. Downstream training uses Adam, validation every 10 epochs, a maximum of 2000 epochs, and patience of 10 validations. CI uses a survival loss and comparable-event ranking pairs; AUC uses binary cross-entropy and death-versus-survival ranking pairs. Both objectives include mean pairwise loss and upper-20% CVaR. AUC denotes observed death-status ROC AUC.

The fixed validation folds are used for checkpoint selection, hyperparameter selection, and five-fold reporting. Reported standard deviations use `ddof=1`. Random seed is 42. `--epochs N` and `--encoder-max-epochs N` explicitly override the training budgets for development runs.

Outputs include selected checkpoints, per-fold patient risk scores, fold metrics, a five-fold summary, and the search results. Pipeline and checkpoint entrypoints also export the four representations. Existing nonempty run directories are rejected.


