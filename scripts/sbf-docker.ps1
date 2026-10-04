# Pass the same arguments as sbf, e.g. .\scripts\sbf-docker.ps1 evaluate mine.
# Use agent names or paths relative to the repository, not Windows absolute paths.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$image = 'shockbench-flow-starter:local'
$sbfArgs = @($args)
if ($sbfArgs.Count -eq 0) { $sbfArgs = @('--help') }

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Install Docker Desktop and start its Linux engine first.'
}
$dockerOs = & docker info --format '{{.OSType}}'
if ($LASTEXITCODE -ne 0 -or $dockerOs -ne 'linux') {
    throw 'Start Docker Desktop with Linux containers, then run this command again.'
}

# Docker reuses dependency layers and its uv cache on subsequent builds.
& docker build --tag $image $repoRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# /opt/venv holds Linux dependencies; the bind mount keeps edits and outputs on Windows.
# The named volume keeps expensive reference calculations between container runs.
& docker run --rm --init `
    --mount "type=bind,source=$repoRoot,target=/workspace" `
    --mount 'type=volume,source=shockbench-flow-reference-cache,target=/cache' `
    $image @sbfArgs
exit $LASTEXITCODE
