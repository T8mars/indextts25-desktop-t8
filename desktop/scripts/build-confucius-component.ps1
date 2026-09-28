param(
    [string]$SourceRoot = "",
    [string]$DestinationRoot = "",
    [switch]$SkipModelHash
)

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
if (-not $SourceRoot) {
    $SourceRoot = if ($env:T8STAR_CONFUCIUS_SOURCE_ROOT) {
        $env:T8STAR_CONFUCIUS_SOURCE_ROOT
    } else {
        "E:\Confucius4-R2T2"
    }
}
if (-not $DestinationRoot) {
    $DestinationRoot = Join-Path $projectRoot "confucius_component"
}
$source = [IO.Path]::GetFullPath($SourceRoot)
$destination = [IO.Path]::GetFullPath($DestinationRoot)
if (-not (Test-Path -LiteralPath (Join-Path $source "r2t2_core\worker.py") -PathType Leaf)) {
    throw "Confucius4-R2T2 source tree is incomplete: $source"
}
if ($destination -eq $source -or $destination.StartsWith($source + [IO.Path]::DirectorySeparatorChar)) {
    throw "Destination must not be inside the source tree."
}
$projectPrefix = $projectRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (-not $destination.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination must stay inside the IndexTTS workspace: $projectRoot"
}

$expectedModels = @{
    "Confucius4-R2T2-Q8_0.gguf" = @{ Size = 1834422208; Sha256 = "151097e43957a19984ea7de66e8144ce69b95039eb31c93da4f58db367e455c3" }
    "mmproj-Confucius4-R2T2-Q8_0.gguf" = @{ Size = 348336544; Sha256 = "8dc2c67e6a0484114928142d098db7ad94ae9f34c78948ef9d37a9678418cb65" }
}
$modelSource = Join-Path $source "models\Confucius4-R2T2-GGUF"
$workerSource = Join-Path $source ".runtime\worker"
$workerConfig = Join-Path $workerSource "pyvenv.cfg"
if (-not (Test-Path -LiteralPath $workerConfig -PathType Leaf)) {
    throw "Confucius worker venv metadata is missing: $workerConfig"
}
$workerHomeLine = Get-Content -LiteralPath $workerConfig |
    Where-Object { $_ -match '^home\s*=' } |
    Select-Object -First 1
$workerBase = if ($workerHomeLine) {
    [IO.Path]::GetFullPath(($workerHomeLine -replace '^home\s*=\s*', '').Trim())
} else {
    ""
}
if (-not $workerBase -or -not (Test-Path -LiteralPath (Join-Path $workerBase "python.exe") -PathType Leaf)) {
    throw "Confucius worker base Python cannot be resolved from: $workerConfig"
}
$workerSitePackages = Join-Path $workerSource "Lib\site-packages"
if (-not (Test-Path -LiteralPath $workerSitePackages -PathType Container)) {
    throw "Confucius worker site-packages are missing: $workerSitePackages"
}
foreach ($entry in $expectedModels.GetEnumerator()) {
    $path = Join-Path $modelSource $entry.Key
    $file = Get-Item -LiteralPath $path
    if ($file.Length -ne $entry.Value.Size) { throw "Unexpected model size: $path" }
    if (-not $SkipModelHash) {
        $stream = [IO.File]::OpenRead($path)
        try {
            $sha256 = [Security.Cryptography.SHA256]::Create()
            try {
                $actual = ([BitConverter]::ToString($sha256.ComputeHash($stream))).Replace("-", "").ToLowerInvariant()
            } finally {
                $sha256.Dispose()
            }
        } finally {
            $stream.Dispose()
        }
        if ($actual -ne $entry.Value.Sha256) { throw "Model SHA-256 mismatch: $path" }
    }
}

$temporary = $destination + ".staging"
if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Recurse -Force }
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    $coreTarget = Join-Path $temporary "r2t2_core"
    New-Item -ItemType Directory -Path $coreTarget -Force | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $source "r2t2_core") -File -Filter "*.py" |
        Copy-Item -Destination $coreTarget
    New-Item -ItemType Directory -Path (Join-Path $temporary ".runtime\build-native-cu128\bin") -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $temporary ".runtime\build-native-cu128\python") -Force | Out-Null
    # A copied venv is not portable on Windows: its pyvenv.cfg keeps absolute
    # paths to the build machine's base interpreter. Build a self-contained
    # Python tree instead, then overlay only the worker's isolated packages.
    $portableWorker = Join-Path $temporary ".runtime\worker"
    New-Item -ItemType Directory -Path $portableWorker -Force | Out-Null
    Get-ChildItem -LiteralPath $workerBase -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $portableWorker -Recurse
    }
    foreach ($name in @("python.exe", "python3.dll", "python312.dll", "vcruntime140.dll")) {
        $runtimeFile = Join-Path $workerBase $name
        if (-not (Test-Path -LiteralPath $runtimeFile -PathType Leaf)) {
            throw "Portable worker base runtime file is missing: $runtimeFile"
        }
        Copy-Item -LiteralPath $runtimeFile -Destination (Join-Path $portableWorker $name) -Force
    }
    if (-not (Test-Path -LiteralPath (Join-Path $portableWorker "python.exe") -PathType Leaf)) {
        throw "Portable worker Python was not copied from: $workerBase"
    }
    $portableSitePackages = Join-Path $portableWorker "Lib\site-packages"
    if (Test-Path -LiteralPath $portableSitePackages) {
        Remove-Item -LiteralPath $portableSitePackages -Recurse -Force
    }
    New-Item -ItemType Directory -Path $portableSitePackages -Force | Out-Null
    Get-ChildItem -LiteralPath $workerSitePackages -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $portableSitePackages -Recurse
    }
    # Keep the source environment's bytecode caches. Recursively pruning
    # __pycache__ directories with Windows PowerShell 5.1 can traverse copied
    # reparse points and remove files outside the intended cache directory.
    # They are harmless in the portable runtime and preserving them is safer.
    Copy-Item -LiteralPath (Join-Path $source ".runtime\build-native-cu128\bin\Release") -Destination (Join-Path $temporary ".runtime\build-native-cu128\bin") -Recurse
    Copy-Item -LiteralPath (Join-Path $source ".runtime\build-native-cu128\python\Release") -Destination (Join-Path $temporary ".runtime\build-native-cu128\python") -Recurse
    $modelTarget = Join-Path $temporary "models\Confucius4-R2T2-GGUF"
    New-Item -ItemType Directory -Path $modelTarget -Force | Out-Null
    foreach ($name in $expectedModels.Keys) {
        Copy-Item -LiteralPath (Join-Path $modelSource $name) -Destination (Join-Path $modelTarget $name)
    }
    foreach ($name in @("MODEL_LICENSE", "MODEL_LICENSE_zh", "README.md")) {
        $metadata = Join-Path $modelSource $name
        if (Test-Path -LiteralPath $metadata -PathType Leaf) {
            Copy-Item -LiteralPath $metadata -Destination (Join-Path $modelTarget $name)
        }
    }
    New-Item -ItemType Directory -Path (Join-Path $temporary "models") -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $source "models\FireRedVAD-ONNX") -Destination (Join-Path $temporary "models") -Recurse
    New-Item -ItemType Directory -Path (Join-Path $temporary "licenses") -Force | Out-Null
    foreach ($name in @("LICENSE", "MODEL_LICENSE", "NOTICE.md")) {
        Copy-Item -LiteralPath (Join-Path $source "vendor\r2t2_native\$name") -Destination (Join-Path $temporary "licenses\$name")
    }
    $manifest = [ordered]@{
        schemaVersion = 1
        component = "Confucius4-R2T2"
        protocolVersion = 1
        profile = "sm120"
        workerPython = ".runtime/worker/python.exe"
        modelDir = "models/Confucius4-R2T2-GGUF"
        buildDir = ".runtime/build-native-cu128"
        modelLicenseSha256 = "064483c5ba1dc20907038108da4f45d5b37cd0a41a351c8a8bb98ba1af48c505"
        models = $expectedModels
    }
    $manifestJson = $manifest | ConvertTo-Json -Depth 5
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText(
        (Join-Path $temporary "confucius-component.json"),
        $manifestJson,
        $utf8NoBom
    )
    if (Test-Path -LiteralPath (Join-Path $portableWorker "pyvenv.cfg")) {
        throw "Portable Confucius worker must not contain pyvenv.cfg."
    }
    $previousCheckRoot = $env:T8_CONFUCIUS_CHECK_ROOT
    $previousPythonNoUserSite = $env:PYTHONNOUSERSITE
    try {
        $env:T8_CONFUCIUS_CHECK_ROOT = $temporary
        $env:PYTHONNOUSERSITE = "1"
        $checkScript = @'
import os
import pathlib
import sys

root = pathlib.Path(os.environ["T8_CONFUCIUS_CHECK_ROOT"]).resolve()
assert root in pathlib.Path(sys.executable).resolve().parents
assert root in pathlib.Path(sys.base_prefix).resolve().parents
sys.path.insert(0, str(root))
from r2t2_core.native import _load_extension
native = _load_extension(root / ".runtime" / "build-native-cu128")
import numpy
import onnxruntime
import soundfile
assert pathlib.Path(native.__file__).resolve().is_relative_to(root)
'@
        # Windows PowerShell 5.1 strips nested quotes from a multiline `-c`
        # argument passed to a native executable. Run the isolated check from
        # a temporary UTF-8 script instead so pathlib string literals survive.
        $checkPath = Join-Path $temporary "verify-portable-worker.py"
        [IO.File]::WriteAllText($checkPath, $checkScript, $utf8NoBom)
        & (Join-Path $portableWorker "python.exe") -I $checkPath
        if ($LASTEXITCODE -ne 0) {
            throw "Portable Confucius worker/native import check failed."
        }
        Remove-Item -LiteralPath $checkPath -Force
    } finally {
        $env:T8_CONFUCIUS_CHECK_ROOT = $previousCheckRoot
        $env:PYTHONNOUSERSITE = $previousPythonNoUserSite
    }
    if (Test-Path -LiteralPath $destination) { Remove-Item -LiteralPath $destination -Recurse -Force }
    Move-Item -LiteralPath $temporary -Destination $destination
} catch {
    if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Recurse -Force }
    throw
}

$bytes = (Get-ChildItem -LiteralPath $destination -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("Confucius component staged at {0} ({1:N2} GiB)." -f $destination, ($bytes / 1GB))
