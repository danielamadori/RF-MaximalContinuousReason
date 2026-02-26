#!/usr/bin/env bash
# ==========================================
# DRIFTS - Docker Management
# ==========================================

set -e

IMAGE_NAME="drifts:latest"
CONTAINER_NAME="drifts-container"

# Parse command
CMD="${1:-start}"

case "$CMD" in
    start)
        echo ""
        echo "=========================================="
        echo "  DRIFTS - Starting Container"
        echo "=========================================="
        echo ""

        # Check Docker
        if ! command -v docker &> /dev/null; then
            echo "[ERROR] Docker not found!"
            echo "Install Docker: https://docs.docker.com/get-docker/"
            exit 1
        fi

        echo "[1/4] Checking Docker... OK"
        echo ""

        # Build image
        echo "[2/4] Building image $IMAGE_NAME..."
        docker build -t "$IMAGE_NAME" .
        echo "Build completed!"
        echo ""

        # Remove existing container
        echo "[3/4] Removing existing container..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
        echo ""

        # Start container
        echo "[4/4] Starting container $CONTAINER_NAME..."
        docker run -d \
            --name "$CONTAINER_NAME" \
            -p 6379:6379 \
            -p 8888:8888 \
            -v "$(pwd)/logs:/app/logs" \
            -v "$(pwd)/workers:/app/workers" \
            -v "$(pwd)/results:/app/results" \
            -v "$(pwd)/fig:/app/fig" \
            "$IMAGE_NAME"

        echo ""
        echo "=========================================="
        echo "  Container started successfully!"
        echo "=========================================="
        echo ""
        echo "Container: $CONTAINER_NAME"
        echo "Redis:     http://localhost:6379"
        echo "Jupyter:   http://localhost:8888"
        echo ""
        echo "Commands:"
        echo "  ./run.sh stop      Stop container"
        echo "  ./run.sh shell     Open shell"
        echo "  ./run.sh logs      View logs"
        echo ""
        ;;

    stop)
        echo ""
        echo "Stopping container $CONTAINER_NAME..."
        docker stop "$CONTAINER_NAME"
        echo "Container stopped."
        echo ""
        ;;

    restart)
        echo ""
        echo "Restarting container $CONTAINER_NAME..."
        docker restart "$CONTAINER_NAME"
        echo "Container restarted."
        echo ""
        ;;

    shell)
        echo ""
        echo "Opening shell in $CONTAINER_NAME..."
        docker exec -it "$CONTAINER_NAME" bash
        ;;

    logs)
        docker logs -f "$CONTAINER_NAME"
        ;;

    rebuild)
        echo ""
        echo "=========================================="
        echo "  DRIFTS - Rebuilding Image"
        echo "=========================================="
        echo ""
        docker build --no-cache -t "$IMAGE_NAME" .
        echo "Build completed!"
        echo ""
        # After rebuild, we usually want to start? Or just build?
        # The bat file goes to start. Let's do the same or just exit?
        # run.bat goes to start.
        "$0" start
        ;;

    clean-rebuild)
        echo ""
        echo "=========================================="
        echo "  DRIFTS - Clean Rebuild"
        echo "=========================================="
        echo ""
        echo "Removing container..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
        "$0" rebuild
        ;;

    help|--help|-h)
        echo ""
        echo "Usage: ./run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  start     Build and start container (default)"
        echo "  stop      Stop container"
        echo "  restart   Restart container"
        echo "  shell     Open bash shell in container"
        echo "  logs      Show container logs"
        echo "  rebuild   Rebuild image (no cache)"
        echo "  clean-rebuild Remove container and rebuild (no cache)"
        echo "  help      Show this help"
        echo ""
        ;;

    *)
        echo "Unknown command: $CMD"
        echo "Run './run.sh help' for usage."
        exit 1
        ;;
esac

