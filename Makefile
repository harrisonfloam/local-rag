.PHONY: help up down build logs clean test restart pull-model

# Default target
help:
	@echo Available commands:
	@echo "  up		- Start the application (shows only app logs)"
	@echo "  down		- Stop all services"
	@echo "  build		- Build the Docker images"
	@echo "  logs		- Show logs from all services"
	@echo "  logs-app	- Show only app logs"
	@echo "  logs-ollama	- Show only ollama logs"
	@echo "  clean		- Remove containers and volumes"
	@echo "  restart	- Restart all services"
	@echo "  pull-model	- Pull a model (usage: make pull-model MODEL=llama3)"
	@echo "  test		- Run tests (when implemented)"

# Start application with only app logs visible
up:
	docker compose up --attach app

# Stop all services
down:
	docker compose down

# Build images
build:
	docker compose build

# Show all logs
logs:
	docker compose logs -f

# Show only app logs
logs-app:
	docker compose logs -f app

# Show only ollama logs
logs-ollama:
	docker compose logs -f ollama

# Clean up everything
clean:
	docker compose down -v
	docker compose rm -f
	docker system prune -f

# Restart services
restart: down up

# Pull a new model
pull-model:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make pull-model MODEL=llama3"; \
		exit 1; \
	fi
	docker compose exec ollama ollama pull $(MODEL)

# Run tests
test:
	@echo "Tests not implemented yet"
	# docker compose exec app python -m pytest

# List available models
list-models:
	docker compose exec ollama ollama list