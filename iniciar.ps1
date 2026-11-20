# Arranca el agente de CVs (Qdrant + backends gratuitos) en Windows.
#
#   .\iniciar.ps1                 -> puerto 8000
#   .\iniciar.ps1 -Puerto 8080
#   .\iniciar.ps1 -Reinstalar     -> borra .venv y reinstala dependencias
#
# Si PowerShell bloquea el script:
#   powershell -ExecutionPolicy Bypass -File .\iniciar.ps1
param(
    [int]$Puerto = 8000,
    [switch]$Reinstalar
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if ($Reinstalar -or -not (Test-Path $py)) {
    if (Test-Path (Join-Path $PSScriptRoot ".venv")) {
        Remove-Item (Join-Path $PSScriptRoot ".venv") -Recurse -Force
    }
    Write-Host "Creando entorno virtual..." -ForegroundColor Cyan
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv .venv
    } else {
        & python -m venv .venv
    }
    if (-not (Test-Path $py)) { throw "No pude crear .venv. Instala Python 3.12+ desde python.org" }
    Write-Host "Instalando dependencias..." -ForegroundColor Cyan
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt
}

# ---- Carga .env (formato KEY=VALOR por linea) ----
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    foreach ($linea in Get-Content $envFile -Encoding UTF8) {
        if ($linea -match '^\s*#' -or $linea -notmatch '=') { continue }
        $k, $v = $linea.Split('=', 2)
        [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim().Trim('"'), "Process")
    }
}

# ---- Resumen y avisos de configuracion ----
$emb = if ($env:EMB_BACKEND) { $env:EMB_BACKEND } elseif ($env:HF_TOKEN) { "hf" } else { "local" }
$llm = if ($env:LLM_BACKEND) { $env:LLM_BACKEND } elseif ($env:HF_TOKEN) { "hf" } else { "ollama" }
$qd = if ($env:QDRANT_URL) { "cloud" } else { "local" }
Write-Host "Embeddings: $emb | LLM: $llm | Qdrant: $qd" -ForegroundColor Cyan

if ($llm -eq "openai" -and -not $env:OPENAI_API_KEY) {
    Write-Host "Aviso: falta OPENAI_API_KEY (Groq/Gemini). Edita .env." -ForegroundColor Yellow
}
if ($llm -eq "hf" -and -not $env:HF_TOKEN) {
    Write-Host "Aviso: LLM_BACKEND=hf pero falta HF_TOKEN." -ForegroundColor Yellow
}
if ($llm -eq "ollama") {
    $ollamaUrl = if ($env:OLLAMA_URL) { $env:OLLAMA_URL } else { "http://127.0.0.1:11434" }
    try {
        Invoke-RestMethod "$ollamaUrl/api/tags" -TimeoutSec 3 | Out-Null
        Write-Host "Ollama responde en $ollamaUrl" -ForegroundColor Green
    } catch {
        Write-Host "Aviso: Ollama no responde. Abre la app de Ollama o ejecuta 'ollama serve'." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Documentacion interactiva: http://127.0.0.1:$Puerto/docs" -ForegroundColor Green
Write-Host "Diagnostico: http://127.0.0.1:$Puerto/salud" -ForegroundColor DarkGray
Write-Host "Pulsa Ctrl+C para parar." -ForegroundColor DarkGray
& $py -m uvicorn main:app --host 127.0.0.1 --port $Puerto
