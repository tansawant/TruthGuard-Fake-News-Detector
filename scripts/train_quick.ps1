# Quick local (CPU) training so the app works right away.
# Baseline uses the full dataset; BERT/RoBERTa use a small balanced subset.
# For full-quality models, use notebooks/Fake_News_BERT_Colab.ipynb on a GPU.
param(
    [int]$Sample = 3000,
    [int]$MaxLen = 128,
    [int]$Epochs = 1,
    [switch]$SkipRoberta
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"

& $py -m src.data
& $py -m src.train_baseline
& $py -m src.train_transformer --model bert --sample $Sample --epochs $Epochs --max-len $MaxLen
if (-not $SkipRoberta) {
    & $py -m src.train_transformer --model roberta --sample $Sample --epochs $Epochs --max-len $MaxLen
}
& $py -m src.compare
Write-Host "`nDone. Start the app with:  .\.venv\Scripts\streamlit run app.py" -ForegroundColor Green
