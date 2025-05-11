# Build script for crawler service
$ErrorActionPreference = "Stop"

# Get the root directory
$rootDir = (Get-Item $PSScriptRoot).Parent.Parent.FullName

# Copy lumina-core to a temporary directory
$tempDir = Join-Path $PSScriptRoot "temp"
if (Test-Path $tempDir) {
    Remove-Item -Recurse -Force $tempDir
}
New-Item -ItemType Directory -Path $tempDir | Out-Null
Copy-Item -Recurse (Join-Path $rootDir "lumina-core") (Join-Path $tempDir "lumina-core")

# Build the Docker image
try {
    docker build -t lumina/crawler:phase9 .
} finally {
    # Clean up temporary directory
    if (Test-Path $tempDir) {
        Remove-Item -Recurse -Force $tempDir
    }
} 