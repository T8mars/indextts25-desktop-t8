param(
  [Parameter(Mandatory = $true)]
  [string]$PayloadRoot,
  [Parameter(Mandatory = $true)]
  [string]$DestinationPath
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $PayloadRoot -PathType Container)) {
  throw "Payload root does not exist."
}
$items = Get-ChildItem -LiteralPath $PayloadRoot -Force | Select-Object -ExpandProperty FullName
if (@($items).Count -eq 0) {
  throw "Payload root is empty."
}
Compress-Archive -LiteralPath $items -DestinationPath $DestinationPath -CompressionLevel Optimal -Force
