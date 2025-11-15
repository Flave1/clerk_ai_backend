#!/bin/bash
# Helper script to manage local DynamoDB

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLERK_BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$CLERK_BACKEND_DIR/docker-compose.yml"

function usage() {
    echo "Usage: $0 {start|stop|restart|status|logs}"
    echo ""
    echo "Commands:"
    echo "  start   - Start local DynamoDB"
    echo "  stop    - Stop local DynamoDB"
    echo "  restart - Restart local DynamoDB"
    echo "  status  - Check if local DynamoDB is running"
    echo "  logs    - Show local DynamoDB logs"
    exit 1
}

function start_dynamodb() {
    echo "Starting local DynamoDB..."
    cd "$CLERK_BACKEND_DIR"
    docker-compose -f "$COMPOSE_FILE" up -d dynamodb
    echo "Waiting for DynamoDB to be ready..."
    sleep 2
    if docker-compose -f "$COMPOSE_FILE" ps dynamodb | grep -q "Up"; then
        echo "✅ Local DynamoDB is running at http://localhost:8001"
    else
        echo "❌ Failed to start local DynamoDB"
        exit 1
    fi
}

function stop_dynamodb() {
    echo "Stopping local DynamoDB..."
    cd "$CLERK_BACKEND_DIR"
    docker-compose -f "$COMPOSE_FILE" stop dynamodb
    echo "✅ Local DynamoDB stopped"
}

function restart_dynamodb() {
    echo "Restarting local DynamoDB..."
    stop_dynamodb
    start_dynamodb
}

function status_dynamodb() {
    cd "$CLERK_BACKEND_DIR"
    if docker-compose -f "$COMPOSE_FILE" ps dynamodb | grep -q "Up"; then
        echo "✅ Local DynamoDB is running at http://localhost:8001"
        docker-compose -f "$COMPOSE_FILE" ps dynamodb
    else
        echo "❌ Local DynamoDB is not running"
        exit 1
    fi
}

function logs_dynamodb() {
    cd "$CLERK_BACKEND_DIR"
    docker-compose -f "$COMPOSE_FILE" logs -f dynamodb
}

case "${1:-}" in
    start)
        start_dynamodb
        ;;
    stop)
        stop_dynamodb
        ;;
    restart)
        restart_dynamodb
        ;;
    status)
        status_dynamodb
        ;;
    logs)
        logs_dynamodb
        ;;
    *)
        usage
        ;;
esac

