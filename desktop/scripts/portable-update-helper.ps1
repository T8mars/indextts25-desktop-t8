param(
  [Parameter(Mandatory = $true)]
  [string]$PlanPath
)

$ErrorActionPreference = "Stop"

function Resolve-FullPath([string]$Value) {
  return [System.IO.Path]::GetFullPath($Value)
}

function Get-Sha256([string]$FilePath) {
  $stream = [System.IO.File]::OpenRead($FilePath)
  try {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
      return ([System.BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    } finally {
      $algorithm.Dispose()
    }
  } finally {
    $stream.Dispose()
  }
}

function Test-PathInside([string]$Candidate, [string]$Root) {
  $rootWithSeparator = $Root.TrimEnd('\') + '\'
  return $Candidate.StartsWith($rootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)
}

function Resolve-RelativeFile([string]$Root, [string]$RelativePath) {
  if ([string]::IsNullOrWhiteSpace($RelativePath) -or
      [System.IO.Path]::IsPathRooted($RelativePath) -or
      $RelativePath.Contains('\') -or
      $RelativePath -match '(^|/)\.\.(/|$)') {
    throw "Unsafe update path: $RelativePath"
  }
  foreach ($segment in @($RelativePath -split '/')) {
    if ([string]::IsNullOrWhiteSpace($segment) -or $segment -match '[<>:"|?*\x00-\x1f]' -or $segment -match '[. ]$') {
      throw "Invalid Windows update path: $RelativePath"
    }
    $stem = ($segment -split '\.')[0].ToUpperInvariant()
    if ($stem -match '^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$') {
      throw "Reserved Windows update path: $RelativePath"
    }
  }
  $candidate = Resolve-FullPath (Join-Path $Root ($RelativePath -replace '/', '\'))
  if (-not (Test-PathInside $candidate $Root)) {
    throw "Update path escapes the allowed root: $RelativePath"
  }
  return $candidate
}

function Write-Result([string]$Status, [string]$Message) {
  $result = [ordered]@{
    status = $Status
    message = $Message
    version = [string]$plan.targetVersion
    completedAt = [DateTime]::UtcNow.ToString('o')
  }
  $json = $result | ConvertTo-Json -Depth 5
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($resultPath, $json, $utf8NoBom)
}

function Restore-Backup {
  $entries = @($script:appliedEntries)
  [array]::Reverse($entries)
  foreach ($entry in $entries) {
    $relative = [string]$entry.path
    $destination = Resolve-RelativeFile $installRoot $relative
    $backup = Resolve-RelativeFile $backupRoot $relative
    if ([bool]$entry.hadOriginal) {
      if (-not (Test-Path -LiteralPath $backup -PathType Leaf)) {
        throw "Backup file is missing: $relative"
      }
      New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
      Copy-Item -LiteralPath $backup -Destination $destination -Force
    } elseif (Test-Path -LiteralPath $destination -PathType Leaf) {
      Remove-Item -LiteralPath $destination -Force
    }
  }
}

$resolvedPlanPath = Resolve-FullPath $PlanPath
if (-not (Test-Path -LiteralPath $resolvedPlanPath -PathType Leaf)) {
  throw "Update plan does not exist: $resolvedPlanPath"
}
$plan = Get-Content -LiteralPath $resolvedPlanPath -Raw -Encoding UTF8 | ConvertFrom-Json
$installRoot = Resolve-FullPath ([string]$plan.installRoot)
$payloadRoot = Resolve-FullPath ([string]$plan.payloadRoot)
$backupRoot = Resolve-FullPath ([string]$plan.backupRoot)
$healthMarker = Resolve-FullPath ([string]$plan.healthMarker)
$executablePath = Resolve-FullPath ([string]$plan.executablePath)
$updatesRoot = Resolve-FullPath ([string]$plan.updatesRoot)
$resultPath = Resolve-FullPath ([string]$plan.resultPath)
$parentPid = [int]$plan.parentPid
$healthToken = [string]$plan.healthToken
$script:appliedEntries = @()

if (-not (Test-Path -LiteralPath $installRoot -PathType Container)) { throw "Install root is missing." }
if (-not (Test-Path -LiteralPath $payloadRoot -PathType Container)) { throw "Payload root is missing." }
if (-not (Test-PathInside $payloadRoot $updatesRoot)) {
  throw "Payload root is outside the updater data directory."
}
if (-not (Test-PathInside $backupRoot $updatesRoot)) {
  throw "Backup root is outside the updater data directory."
}
if (-not (Test-PathInside $healthMarker $updatesRoot)) {
  throw "Health marker is outside the updater data directory."
}
if (-not (Test-PathInside $resultPath $updatesRoot)) { throw "Result path is outside the updater data directory." }
if (-not (Test-PathInside $executablePath $installRoot)) { throw "Executable is outside the install root." }
if ($healthToken -notmatch '^[a-f0-9-]{16,64}$') { throw "Health token is invalid." }
if (@($plan.files).Count -eq 0) { throw "Update plan has no files." }

try {
  Wait-Process -Id $parentPid -Timeout 120 -ErrorAction SilentlyContinue
  Start-Sleep -Milliseconds 500
  New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null

  foreach ($entry in @($plan.files)) {
    $relative = [string]$entry.path
    $source = Resolve-RelativeFile $payloadRoot $relative
    $destination = Resolve-RelativeFile $installRoot $relative
    $backup = Resolve-RelativeFile $backupRoot $relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
      throw "Payload file is missing: $relative"
    }
    $expectedSize = [long]$entry.size
    $expectedSha256 = ([string]$entry.sha256).ToLowerInvariant()
    if ($expectedSize -lt 0 -or $expectedSha256 -notmatch '^[a-f0-9]{64}$') {
      throw "Payload metadata is invalid: $relative"
    }
    if ((Get-Item -LiteralPath $source).Length -ne $expectedSize) {
      throw "Payload file size changed before installation: $relative"
    }
    if ((Get-Sha256 $source) -ne $expectedSha256) {
      throw "Payload file hash changed before installation: $relative"
    }
    $hadOriginal = Test-Path -LiteralPath $destination -PathType Leaf
    if ($hadOriginal) {
      New-Item -ItemType Directory -Force -Path (Split-Path -Parent $backup) | Out-Null
      Copy-Item -LiteralPath $destination -Destination $backup -Force
    }
    $script:appliedEntries += [pscustomobject]@{ path = $relative; hadOriginal = $hadOriginal }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
    if ((Get-Item -LiteralPath $destination).Length -ne $expectedSize -or
        (Get-Sha256 $destination) -ne $expectedSha256) {
      throw "Installed file verification failed: $relative"
    }
  }

  Remove-Item -LiteralPath $healthMarker -Force -ErrorAction SilentlyContinue
  $updatedProcess = Start-Process -FilePath $executablePath -ArgumentList @(
    '--update-token', $healthToken,
    '--update-health-marker', $healthMarker
  ) -PassThru

  $healthy = $false
  $deadline = [DateTime]::UtcNow.AddSeconds(90)
  while ([DateTime]::UtcNow -lt $deadline) {
    if (Test-Path -LiteralPath $healthMarker -PathType Leaf) {
      $reportedToken = (Get-Content -LiteralPath $healthMarker -Raw -Encoding UTF8).Trim()
      if ($reportedToken -eq $healthToken) {
        $healthy = $true
        break
      }
    }
    if ($updatedProcess.HasExited) { break }
    Start-Sleep -Milliseconds 500
    $updatedProcess.Refresh()
  }

  if (-not $healthy) {
    if (-not $updatedProcess.HasExited) {
      Stop-Process -Id $updatedProcess.Id -Force -ErrorAction SilentlyContinue
      Start-Sleep -Milliseconds 500
    }
    Restore-Backup
    Write-Result 'rolled-back' 'The updated app did not pass its startup health check. The previous version was restored automatically.'
    Start-Process -FilePath $executablePath -ArgumentList @('--update-rollback') | Out-Null
    exit 2
  }

  Write-Result 'installed' 'The update was installed and passed its startup health check.'
  exit 0
} catch {
  try { Restore-Backup } catch { }
  try { Write-Result 'failed' ([string]$_.Exception.Message) } catch { }
  try { Start-Process -FilePath $executablePath -ArgumentList @('--update-rollback') | Out-Null } catch { }
  exit 1
}
