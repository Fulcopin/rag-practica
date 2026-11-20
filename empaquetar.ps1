# Crea un ZIP con lo necesario para llevar el proyecto a otra computadora.
#
#   .\empaquetar.ps1                  -> agente_cv.zip (sin CVs indexados)
#   .\empaquetar.ps1 -ConDatos        -> incluye qdrant_data (tus CVs ya indexados)
#   .\empaquetar.ps1 -ConModelos      -> incluye models_cache (240 MB, evita descargar)
#   .\empaquetar.ps1 -Destino D:\x.zip
param(
    [string]$Destino = (Join-Path $PSScriptRoot "agente_cv.zip"),
    [switch]$ConDatos,
    [switch]$ConModelos
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$tmp = Join-Path $env:TEMP ("empaquetar_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp | Out-Null

$items = @("main.py", "panel.html", "requirements.txt", "probar_agente.py", "ver_qdrant.py", "subir_cvs.py",
           "iniciar.ps1", "INSTALAR.md", ".env.example", "agente_cv_qdrant.ipynb", ".env")
foreach ($i in $items) {
    if (Test-Path $i) { Copy-Item $i -Destination $tmp }
}
if (Test-Path "cvs_ejemplo") { Copy-Item "cvs_ejemplo" -Destination $tmp -Recurse }
if ($ConDatos -and (Test-Path "qdrant_data")) {
    Copy-Item "qdrant_data" -Destination $tmp -Recurse
    Write-Host "Incluyendo qdrant_data (CVs ya indexados)." -ForegroundColor Cyan
}
if ($ConModelos -and (Test-Path "models_cache")) {
    Copy-Item "models_cache" -Destination $tmp -Recurse
    Write-Host "Incluyendo models_cache (modelo de embeddings)." -ForegroundColor Cyan
}

if (Test-Path $Destino) { Remove-Item $Destino -Force }
Compress-Archive -Path (Join-Path $tmp "*") -DestinationPath $Destino
Remove-Item $tmp -Recurse -Force

Write-Host "Listo: $Destino" -ForegroundColor Green
if (Test-Path (Join-Path $PSScriptRoot ".env")) {
    Write-Host "Ojo: el ZIP incluye .env (tiene tus claves de Groq/Qdrant). No lo compartas." -ForegroundColor Yellow
}
