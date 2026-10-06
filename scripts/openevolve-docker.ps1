param(
    [ValidateSet('constant', 'adaptive', 'mpc')][string]$Strategy = 'constant',
    [ValidateSet('tiny', 'small', 'full', 'mixed')][string]$Task = 'small',
    [int]$Episodes = 8,
    [int]$Iterations = 10,
    [long]$Entropy = 202610051,
    [int]$Jobs = 2,
    [string]$Model = 'qwen2.5-coder:3b',
    [string]$Out = '',
    [string]$Seed = '',
    [switch]$Warmup,
    [double]$Temperature = 0.95,
    [int]$ProposalAttempts = 4,
    [double]$MinChange = 0.03,
    [int]$Context = 8192,
    [switch]$Rebuild
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$image = 'shockbench-flow-openevolve:local'
if ([Math]::Min($Episodes, [Math]::Min($Iterations, $Jobs)) -lt 1 -or $Entropy -le 0) {
    throw 'Episodes, Iterations, Jobs and Entropy must be positive.'
}
$dockerOs = & docker info --format '{{.OSType}}'
if ($LASTEXITCODE -ne 0 -or $dockerOs -ne 'linux') {
    throw 'Start Docker Desktop with Linux containers first.'
}
$baseImageId = & docker image ls --quiet 'shockbench-flow-starter:local'
if (-not $baseImageId) {
    & docker build --tag 'shockbench-flow-starter:local' $repoRoot
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$searchImageId = & docker image ls --quiet $image
if ($Rebuild -or -not $searchImageId) {
    & docker build --file "$repoRoot\scripts\Dockerfile.openevolve" --tag $image $repoRoot
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$runArgs = @("--strategy=$Strategy", "--task=$Task", "--episodes=$Episodes", "--iterations=$Iterations", "--entropy=$Entropy",
    "--n_jobs=$Jobs", "--model=$Model", "--temperature=$($Temperature.ToString([cultureinfo]::InvariantCulture))",
    "--proposal_attempts=$ProposalAttempts", "--min_change=$($MinChange.ToString([cultureinfo]::InvariantCulture))",
    "--context=$Context")
if ($Out) { $runArgs += "--out=$Out" }
if ($Seed) { $runArgs += "--seed=$Seed" }
if ($Warmup) { $runArgs += '--warmup=True' }
& docker run --rm --init --cpus=3 --memory=8g `
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 `
    --mount "type=bind,source=$repoRoot,target=/workspace" `
    --mount 'type=volume,source=shockbench-flow-reference-cache,target=/cache' `
    $image @runArgs
exit $LASTEXITCODE
