param(
    [switch]$IncludeModels
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$sourceArchive = Join-Path $projectRoot "ptq_server_source_$stamp.tar.gz"
$sourcePaths = @(
    'ptq',
    'experiments/revision_full',
    'experiments/fix_gsm8k_500/direct_eval.py',
    'experiments/fix_svamp_ood',
    'requirements-server.txt',
    'README.md'
)

Push-Location $projectRoot
try {
    & tar.exe -czf $sourceArchive --exclude='__pycache__' --exclude='*.pyc' @sourcePaths
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create $sourceArchive"
    }
    Write-Output $sourceArchive

    if ($IncludeModels) {
        $modelArchive = Join-Path $projectRoot "ptq_server_models_$stamp.tar"
        $modelPaths = @(
            'models/Qwen2.5-0.5B',
            'models/Qwen2.5-1.5B',
            'models/SmolLM-1.7B',
            'models/gemma-2-2b-it'
        )
        & tar.exe -cf $modelArchive @modelPaths
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create $modelArchive"
        }
        Write-Output $modelArchive
    }
}
finally {
    Pop-Location
}
