# CiRA ME - Deployment Helper Script (PowerShell)
# Makes it easy to deploy and manage the CiRA ME application on Windows

param(
    [Parameter(Position=0)]
    [string]$Command = "help",

    [Parameter(Position=1)]
    [string]$Argument = ""
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."

    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Info "Docker found: $dockerVersion"
    }
    catch {
        Write-Error-Custom "Docker is not installed. Please install Docker Desktop for Windows."
        exit 1
    }

    # Check Docker Compose
    try {
        $composeVersion = docker compose version
        Write-Info "Docker Compose found: $composeVersion"
    }
    catch {
        Write-Error-Custom "Docker Compose is not installed. Please install Docker Compose."
        exit 1
    }

    Write-Info "Prerequisites check passed."
}

# Function to create required directories
function New-RequiredDirectories {
    Write-Info "Creating required directories..."

    $directories = @(
        "docker-volumes\data",
        "docker-volumes\models",
        "docker-volumes\datasets"
    )

    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -Path $dir -ItemType Directory -Force | Out-Null
            Write-Info "Created directory: $dir"
        }
        else {
            Write-Info "Directory already exists: $dir"
        }
    }

    Write-Info "Directories created successfully."
}

# Function to setup environment
function Initialize-Environment {
    if (-not (Test-Path ".env")) {
        Write-Info "Creating .env file from .env.example..."
        Copy-Item ".env.example" ".env"
        Write-Warning-Custom "Please review and update .env file with your configuration."
    }
    else {
        Write-Info ".env file already exists. Skipping creation."
    }
}

# Function to build images
function Build-Images {
    Write-Info "Building Docker images..."
    docker compose build --no-cache

    if ($LASTEXITCODE -eq 0) {
        Write-Info "Images built successfully."
    }
    else {
        Write-Error-Custom "Failed to build images."
        exit 1
    }
}

# Function to start services
function Start-Services {
    param([string]$Mode = "dev")

    if ($Mode -eq "prod") {
        Write-Info "Starting services in PRODUCTION mode..."
        docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    }
    else {
        Write-Info "Starting services in DEVELOPMENT mode..."
        docker compose up -d
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Info "Services started successfully."
    }
    else {
        Write-Error-Custom "Failed to start services."
        exit 1
    }
}

# Function to stop services
function Stop-Services {
    Write-Info "Stopping services..."
    docker compose down

    if ($LASTEXITCODE -eq 0) {
        Write-Info "Services stopped successfully."
    }
    else {
        Write-Error-Custom "Failed to stop services."
        exit 1
    }
}

# Function to view logs
function Show-Logs {
    param([string]$Service = "")

    if ($Service -eq "") {
        docker compose logs -f
    }
    else {
        docker compose logs -f $Service
    }
}

# Function to check status
function Get-Status {
    Write-Info "Service status:"
    docker compose ps

    Write-Host ""
    Write-Info "Health checks:"

    try {
        $backendHealth = docker inspect --format='{{.State.Health.Status}}' cirame-backend 2>$null
        Write-Host "Backend: $backendHealth"
    }
    catch {
        Write-Host "Backend: Not running"
    }

    try {
        $frontendHealth = docker inspect --format='{{.State.Health.Status}}' cirame-frontend 2>$null
        Write-Host "Frontend: $frontendHealth"
    }
    catch {
        Write-Host "Frontend: Not running"
    }

    Write-Host ""
    Write-Info "Resource usage:"
    docker stats --no-stream
}

# Function to backup data
function Backup-Data {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupDir = "backups\$timestamp"

    Write-Info "Creating backup in $backupDir..."

    # Create backup directory
    New-Item -Path $backupDir -ItemType Directory -Force | Out-Null

    # Copy data
    Copy-Item -Path "docker-volumes\*" -Destination $backupDir -Recurse -Force

    # Compress backup
    $backupArchive = "$backupDir.zip"
    Compress-Archive -Path "$backupDir\*" -DestinationPath $backupArchive -Force
    Remove-Item -Path $backupDir -Recurse -Force

    Write-Info "Backup created: $backupArchive"
}

# Function to restore data
function Restore-Data {
    param([string]$BackupFile)

    if ($BackupFile -eq "") {
        Write-Error-Custom "Please specify backup file to restore."
        exit 1
    }

    if (-not (Test-Path $BackupFile)) {
        Write-Error-Custom "Backup file not found: $BackupFile"
        exit 1
    }

    Write-Warning-Custom "This will overwrite existing data. Are you sure? (yes/no)"
    $confirmation = Read-Host

    if ($confirmation -ne "yes") {
        Write-Info "Restore cancelled."
        exit 0
    }

    Write-Info "Stopping services..."
    docker compose down

    Write-Info "Restoring from backup..."
    Expand-Archive -Path $BackupFile -DestinationPath "docker-volumes\" -Force

    Write-Info "Starting services..."
    docker compose up -d

    Write-Info "Restore completed successfully."
}

# Function to clean up
function Remove-All {
    Write-Warning-Custom "This will remove all containers, volumes, and images. Are you sure? (yes/no)"
    $confirmation = Read-Host

    if ($confirmation -ne "yes") {
        Write-Info "Cleanup cancelled."
        exit 0
    }

    Write-Info "Cleaning up..."
    docker compose down -v
    docker system prune -a -f

    Write-Info "Cleanup completed."
}

# Function to update application
function Update-Application {
    Write-Info "Updating CiRA ME application..."

    Write-Info "Pulling latest changes (if git repo)..."
    try {
        git pull 2>$null
    }
    catch {
        Write-Warning-Custom "Not a git repository or git not available."
    }

    Write-Info "Backing up current data..."
    Backup-Data

    Write-Info "Rebuilding images..."
    docker compose build --no-cache

    Write-Info "Restarting services..."
    docker compose down
    docker compose up -d

    Write-Info "Update completed successfully."
}

# Function to display usage
function Show-Usage {
    Write-Host @"
CiRA ME - Deployment Helper Script (PowerShell)

Usage: .\deploy.ps1 [command] [options]

Commands:
    setup           Initial setup (check prerequisites, create directories, setup environment)
    build           Build Docker images
    start [mode]    Start services (mode: dev or prod, default: dev)
    stop            Stop services
    restart [mode]  Restart services (mode: dev or prod, default: dev)
    logs [service]  View logs (service: backend or frontend, default: all)
    status          Check service status and health
    backup          Backup application data
    restore <file>  Restore from backup file
    update          Update application (pull changes, rebuild, restart)
    clean           Remove all containers, volumes, and images
    help            Display this help message

Examples:
    .\deploy.ps1 setup                    # Initial setup
    .\deploy.ps1 start dev                # Start in development mode
    .\deploy.ps1 start prod               # Start in production mode
    .\deploy.ps1 logs backend             # View backend logs
    .\deploy.ps1 backup                   # Create backup
    .\deploy.ps1 restore backups\20240101_120000.zip  # Restore from backup

"@
}

# Main script logic
switch ($Command.ToLower()) {
    "setup" {
        Test-Prerequisites
        New-RequiredDirectories
        Initialize-Environment
        Write-Info "Setup completed. Run '.\deploy.ps1 build' to build images, then '.\deploy.ps1 start' to start services."
    }
    "build" {
        Test-Prerequisites
        Build-Images
    }
    "start" {
        Test-Prerequisites
        $mode = if ($Argument -eq "") { "dev" } else { $Argument }
        Start-Services -Mode $mode
        Write-Info "CiRA ME is running at http://localhost:3030"
    }
    "stop" {
        Stop-Services
    }
    "restart" {
        Test-Prerequisites
        $mode = if ($Argument -eq "") { "dev" } else { $Argument }
        Stop-Services
        Start-Services -Mode $mode
    }
    "logs" {
        Show-Logs -Service $Argument
    }
    "status" {
        Get-Status
    }
    "backup" {
        Backup-Data
    }
    "restore" {
        Restore-Data -BackupFile $Argument
    }
    "update" {
        Update-Application
    }
    "clean" {
        Remove-All
    }
    "help" {
        Show-Usage
    }
    default {
        Write-Error-Custom "Unknown command: $Command"
        Show-Usage
        exit 1
    }
}
