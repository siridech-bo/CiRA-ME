#!/bin/bash
# CiRA ME - View Logs Script
# Shows live logs from all containers.
# Press Ctrl+C to exit.
#
# Usage:  bash logs.sh              # follow all CiRA ME container logs
#         bash logs.sh backend      # follow only one service
#         bash logs.sh --no-follow  # dump recent logs and exit

FOLLOW="-f"
SERVICE=""
for arg in "$@"; do
    case "$arg" in
        --no-follow|-n) FOLLOW="" ;;
        backend|frontend|ti-modelmaker|mosquitto) SERVICE="$arg" ;;
    esac
done

echo "============================================"
echo "  CiRA ME - Application Logs"
if [ -n "$FOLLOW" ]; then
    echo "  Press Ctrl+C to exit"
fi
echo "============================================"
echo

# Fail early if nothing is running — better than a silent empty tail.
if ! docker ps --format '{{.Names}}' | grep -q '^cirame-'; then
    echo "ERROR: no CiRA ME containers are running."
    echo "       Start the app first: bash start.sh (or bash start-no-gpu.sh)"
    exit 1
fi

# Pick whichever compose file this deployment is using — we run whichever
# is available first, and skip errors from the other.
DOCKER_COMPOSE=""
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
else
    # Fallback: use raw docker logs on named containers.
    if [ -n "$SERVICE" ]; then
        docker logs $FOLLOW "cirame-$SERVICE" 2>&1
    else
        echo "docker compose not installed — falling back to per-container logs"
        for name in cirame-backend cirame-frontend cirame-ti-modelmaker cirame-mosquitto; do
            docker ps --format '{{.Names}}' | grep -q "^$name$" && \
                docker logs --tail 20 "$name" 2>&1 | sed "s|^|[$name] |"
        done
    fi
    exit 0
fi

for compose in docker-compose.yml docker-compose-no-gpu.yml; do
    if [ -f "$compose" ] && $DOCKER_COMPOSE -f "$compose" ps -q 2>/dev/null | grep -q .; then
        $DOCKER_COMPOSE -f "$compose" logs $FOLLOW $SERVICE
        exit $?
    fi
done

echo "ERROR: docker compose files present, but no services are running under them."
echo "       (containers may have been started outside of compose — try:"
echo "        docker logs $FOLLOW cirame-backend)"
exit 1
