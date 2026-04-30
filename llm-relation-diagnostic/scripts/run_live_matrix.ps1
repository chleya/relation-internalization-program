param(
    [string]$Config = "configs/budgeted_inspect_stress.yaml",
    [string[]]$Models = @("qwen05b", "qwen15b"),
    [int]$Port = 8083,
    [int]$Context = 4096,
    [int]$GpuLayers = 0,
    [int]$ReadyTimeoutSec = 120,
    [string]$RunLabel = "",
    [string]$Server = "F:\AI_Workspace\llama.cpp\build\bin\Release\llama-server.exe"
)

$ErrorActionPreference = "Stop"

function Quote-Arg {
    param([string]$Value)
    if ($Value -match '[\s()]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

function Join-Args {
    param([string[]]$Values)
    return ($Values | ForEach-Object { Quote-Arg $_ }) -join " "
}

function Get-Run-Label {
    param([string]$ConfigPath, [string]$Override)
    if ($Override) {
        return $Override
    }
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($ConfigPath)
    if ($stem -eq "budgeted_inspect_stress") {
        return "budgeted_stress"
    }
    return $stem
}

$modelCatalog = @{
    qwen05b = @{
        ModelName = "qwen2.5-0.5b-instruct-q4_k_m"
        Path = "F:\tmp\models\qwen2.5-0.5b-instruct-q4_k_m.gguf"
    }
    qwen15b = @{
        ModelName = "qwen2.5-1.5b-instruct-q4_k_m"
        Path = "F:\unified-sel-artifacts\gguf_models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
    }
    qwen3b = @{
        ModelName = "qwen2.5-3b-instruct-q5_k_m"
        Path = "F:\unified-sel-artifacts\qwen2.5-3b-instruct-q5_k_m.gguf"
    }
    gemma3_4b = @{
        ModelName = "gemma-3-4b-it-q5_k_m"
        Path = "F:\unified-sel-artifacts\google_gemma-3-4b-it-Q5_K_M (1).gguf"
    }
    phi4mini = @{
        ModelName = "phi-4-mini-instruct-q4_k_m"
        Path = "F:\unified-sel-artifacts\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
    }
}

if (-not (Test-Path -LiteralPath $Server)) {
    throw "llama-server not found: $Server"
}

New-Item -ItemType Directory -Force -Path "results" | Out-Null

$selectedModels = @()
foreach ($item in $Models) {
    $selectedModels += ($item -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}
$resolvedRunLabel = Get-Run-Label -ConfigPath $Config -Override $RunLabel

foreach ($modelKey in $selectedModels) {
    if (-not $modelCatalog.ContainsKey($modelKey)) {
        throw "Unknown model key: $modelKey. Known keys: $($modelCatalog.Keys -join ', ')"
    }

    $entry = $modelCatalog[$modelKey]
    $modelPath = $entry.Path
    $label = "${modelKey}_${resolvedRunLabel}"
    $modelName = $entry.ModelName
    $outLog = "results\llama_server_$label.out.log"
    $errLog = "results\llama_server_$label.err.log"
    $proc = $null

    Write-Host "=== Running $modelKey ($modelPath) ==="
    try {
        if (-not (Test-Path -LiteralPath $modelPath)) {
            throw "model not found: $modelPath"
        }

        $argsLine = Join-Args @("-m", $modelPath, "--port", "$Port", "-c", "$Context", "-ngl", "$GpuLayers")
        $proc = Start-Process -FilePath $Server -ArgumentList $argsLine -WorkingDirectory (Get-Location) -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru

        $ready = $false
        for ($i = 0; $i -lt $ReadyTimeoutSec; $i++) {
            Start-Sleep -Seconds 1
            try {
                $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2
                if ($resp.StatusCode -eq 200) {
                    $ready = $true
                    break
                }
            } catch {
            }
            if ($proc.HasExited) {
                throw "llama-server exited early. See $errLog"
            }
        }

        if (-not $ready) {
            throw "llama-server did not become ready. See $errLog"
        }

        python -m src.run_experiment --config $Config --solvers llama_cpp --base-url "http://127.0.0.1:$Port" --model $modelName --label $label
    } catch {
        $failurePath = "results\live_matrix_failures.csv"
        if (-not (Test-Path -LiteralPath $failurePath)) {
            "model,label,config,error" | Set-Content -Path $failurePath -Encoding utf8
        }
        $escaped = ($_.Exception.Message -replace '"', '""')
        """$modelKey"",""$label"",""$Config"",""$escaped""" | Add-Content -Path $failurePath -Encoding utf8
        Write-Warning "Failed ${modelKey}: $($_.Exception.Message)"
    } finally {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force
        }
    }
}
