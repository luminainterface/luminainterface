.PHONY: up down logs help

help:           ## Show this help message
	@echo 'Usage:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

up:            ## kindly start all back-end services
	docker compose pull
	docker compose up -d
	@echo "🙏  All services are starting—run 'make logs' to watch health checks."

down:          ## gracefully stop everything
	docker compose down

logs:          ## follow aggregated logs
	docker compose logs -f --tail=50 