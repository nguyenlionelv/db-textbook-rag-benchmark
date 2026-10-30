param(
    [ValidateRange(1024, 65535)][int]$Port = 11434,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$ragRoot = Split-Path -Parent $PSScriptRoot
$ragRuntime = Join-Path $ragRoot 'output/model_cache/ollama_runtime'
$ragExecutable = Join-Path $ragRuntime 'ollama.exe'
$ragPidFile = Join-Path $ragRuntime 'server.pid'
if (Test-Path -LiteralPath $ragPidFile) {
    $ragRecordedId = [int](Get-Content -LiteralPath $ragPidFile -Raw)
    $ragExisting = Get-Process -Id $ragRecordedId -ErrorAction SilentlyContinue
    if ($ragExisting -and $ragExisting.Path -eq $ragExecutable) {
        if (-not $Stop) { throw 'Ollama của dự án đang chạy. Dùng -Stop để dừng.' }
        & taskkill.exe /PID $ragRecordedId /T /F | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Không dừng được process Ollama của dự án.' }
    }
    elseif ($Stop -and $ragExisting) {
        throw 'PID không thuộc Ollama của dự án; không dừng process này.'
    }
    Remove-Item -LiteralPath $ragPidFile -Force -ErrorAction SilentlyContinue
}
if ($Stop) { Write-Output 'Ollama của dự án đã dừng.'; return }
if (-not (Test-Path -LiteralPath $ragExecutable)) {
    throw 'Chưa có Ollama CLI tại output/model_cache/ollama_runtime/ollama.exe.'
}

# Pass settings only to Ollama; leave the Windows environment unchanged.
# Go uses USERPROFILE to locate its home directory on Windows.
$ragProfile = Join-Path $ragRuntime 'profile'
$ragTemp = Join-Path $ragRuntime 'tmp'
$ragLocal = Join-Path $ragProfile 'AppData/Local'
$ragRoaming = Join-Path $ragProfile 'AppData/Roaming'
foreach ($ragDirectory in @($ragProfile, $ragTemp, $ragLocal, $ragRoaming)) {
    New-Item -ItemType Directory -Path $ragDirectory -Force | Out-Null
}
$ragChildOptions = @{
    USERPROFILE = $ragProfile
    LOCALAPPDATA = $ragLocal
    APPDATA = $ragRoaming
    TEMP = $ragTemp
    TMP = $ragTemp
    OLLAMA_HOST = "127.0.0.1:$Port"
    OLLAMA_MODELS = (Join-Path $ragRoot 'output/model_cache/ollama_models')
    OLLAMA_NO_CLOUD = '1'
    OLLAMA_NUM_PARALLEL = '1'
    OLLAMA_MAX_LOADED_MODELS = '1'
    OLLAMA_KEEP_ALIVE = '2m'
    OLLAMA_VULKAN = '0'
    OLLAMA_IGPU_ENABLE = '0'
}
$ragStart = [Diagnostics.ProcessStartInfo]::new()
$ragStart.FileName = $ragExecutable
$ragStart.Arguments = 'serve'
$ragStart.WorkingDirectory = $ragRoot
$ragStart.UseShellExecute = $false
$ragStart.CreateNoWindow = $true
foreach ($ragName in $ragChildOptions.Keys) {
    $ragStart.EnvironmentVariables[$ragName] = $ragChildOptions[$ragName]
}
$ragServer = [Diagnostics.Process]::Start($ragStart)
$ragServer.Id | Set-Content -LiteralPath $ragPidFile -Encoding ascii
Write-Output "Ollama PID=$($ragServer.Id), URL=http://127.0.0.1:$Port"
try {
    while (-not $ragServer.WaitForExit(500)) { }
    if ($ragServer.ExitCode -ne 0) { throw "Ollama kết thúc với exit code $($ragServer.ExitCode)" }
}
finally {
    if (-not $ragServer.HasExited) {
        # Stop only the Ollama process tree started by this script.
        & taskkill.exe /PID $ragServer.Id /T /F | Out-Null
    }
    $ragFinalId = Get-Content -LiteralPath $ragPidFile -Raw -ErrorAction SilentlyContinue
    if ($ragFinalId -and [int]$ragFinalId -eq $ragServer.Id) {
        Remove-Item -LiteralPath $ragPidFile -Force -ErrorAction SilentlyContinue
    }
    $ragServer.Dispose()
}
