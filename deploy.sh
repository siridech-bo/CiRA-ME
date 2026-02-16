#!/bin/bash
# CiRA ME - Deployment Helper Script
# Makes it easy to deploy and manage the CiRA ME application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    print_info "Prerequisites check passed."
}

# Function to create required directories
create_directories() {
    print_info "Creating required directories..."

    mkdir -p docker-volumes/data
    mkdir -p docker-volumes/models
    mkdir -p docker-volumes/datasets

    print_info "Directories created successfully."
}

# Function to setup environment
setup_environment() {
    if [ ! -f ".env" ]; then
        print_info "Creating .env file from .env.example..."
        cp .env.example .env
        print_warning "Please review and update .env file with your configuration."
    else
        print_info ".env file already exists. Skipping creation."
    fi
}

# Function to build images
build_images() {
    print_info "Building Docker images..."
    docker compose build --no-cache
    print_info "Images built successfully."
}

# Function to start services
start_services() {
    local mode=${1:-dev}

    if [ "$mode" = "prod" ]; then
        print_info "Starting services in PRODUCTION mode..."
        docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    else
        print_info "Starting services in DEVELOPMENT mode..."
        docker compose up -d
    fi

    print_info "Services started successfully."
}

# Function to stop services
stop_services() {
    print_info "Stopping services..."
    docker compose down
    print_info "Services stopped successfully."
}

# Function to view logs
view_logs() {
    local service=${1:-}

    if [ -z "$service" ]; then
        docker compose logs -f
    else
        docker compose logs -f "$service"
    fi
}

# Function to check status
check_status() {
    print_info "Service status:"
    docker compose ps

    echo ""
    print_info "Health checks:"
    docker inspect --format='Backend: {{.State.Health.Status}}' cirame-backend 2>/dev/null || echo "Backend: Not running"
    docker inspect --format='Frontend: {{.State.Health.Status}}' cirame-frontend 2>/dev/null || echo "Frontend: Not running"

    echo ""
    print_info "Resource usage:"
    docker stats --no-stream
}

# Function to backup data
backup_data() {
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"

    print_info "Creating backup in $backup_dir..."

    mkdir -p "$backup_dir"
    cp -r docker-volumes/* "$backup_dir/"

    tar -czf "${backup_dir}.tar.gz" -C "$backup_dir" .
    rm -rf "$backup_dir"

    print_info "Backup created: ${backup_dir}.tar.gz"
}

# Function to restore data
restore_data() {
    local backup_file=$1

    if [ -z "$backup_file" ]; then
        print_error "Please specify backup file to restore."
        exit 1
    fi

    if [ ! -f "$backup_file" ]; then
        print_error "Backup file not found: $backup_file"
        exit 1
    fi

    print_warning "This will overwrite existing data. Are you sure? (yes/no)"
    read -r confirmation

    if [ "$confirmation" != "yes" ]; then
        print_info "Restore cancelled."
        exit 0
    fi

    print_info "Stopping services..."
    docker compose down

    print_info "Restoring from backup..."
    tar -xzf "$backup_file" -C docker-volumes/

    print_info "Starting services..."
    docker compose up -d

    print_info "Restore completed successfully."
}

# Function to clean up
cleanup() {
    print_warning "This will remove all containers, volumes, and images. Are you sure? (yes/no)"
    read -r confirmation

    if [ "$confirmation" != "yes" ]; then
        print_info "Cleanup cancelled."
        exit 0
    fi

    print_info "Cleaning up..."
    docker compose down -v
    docker system prune -a -f

    print_info "Cleanup completed."
}

# Function to update application
update_application() {
    print_info "Updating CiRA ME application..."

    print_info "Pulling latest changes (if git repo)..."
    git pull 2>/dev/null || print_warning "Not a git repository or git not available."

    print_info "Backing up current data..."
    backup_data

    print_info "Rebuilding images..."
    docker compose build --no-cache

    print_info "Restarting services..."
    docker compose down
    docker compose up -d

    print_info "Update completed successfully."
}

# Function to display usage
usage() {
    cat << EOF
CiRA ME - Deployment Helper Script

Usage: $0 [command] [options]

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
    $0 setup                    # Initial setup
    $0 start dev                # Start in development mode
    $0 start prod               # Start in production mode
    $0 logs backend             # View backend logs
    $0 backup                   # Create backup
    $0 restore backups/20240101_120000.tar.gz  # Restore from backup

EOF
}

# Main script logic
main() {
    local command=${1:-help}

    case $command in
        setup)
            check_prerequisites
            create_directories
            setup_environment
            print_info "Setup completed. Run '$0 build' to build images, then '$0 start' to start services."
            ;;
        build)
            check_prerequisites
            build_images
            ;;
        start)
            check_prerequisites
            local mode=${2:-dev}
            start_services "$mode"
            print_info "CiRA ME is running at http://localhost:3030"
            ;;
        stop)
            stop_services
            ;;
        restart)
            check_prerequisites
            local mode=${2:-dev}
            stop_services
            start_services "$mode"
            ;;
        logs)
            view_logs "$2"
            ;;
        status)
            check_status
            ;;
        backup)
            backup_data
            ;;
        restore)
            restore_data "$2"
            ;;
        update)
            update_application
            ;;
        clean)
            cleanup
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            print_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
