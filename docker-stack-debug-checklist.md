# Docker Stack Debug Checklist

Use this checklist to systematically debug and bring up your full stack as shown in your architecture diagram and docker-compose.yml.

## 1. Core Infrastructure
- [ ] Redis container is running and healthy
- [ ] Qdrant container is running and healthy
- [ ] Neo4j container is running and healthy

## 2. Observability & Monitoring
- [ ] Prometheus is running and scraping targets
- [ ] Grafana is running and dashboards are accessible
- [ ] Loki is running and logs are being collected
- [ ] Jaeger is running and traces are being collected
- [ ] Alertmanager is running
- [ ] Redis Exporter is running
- [ ] Promtail is running

## 3. Core Services
- [ ] Concept-Dict is running and healthy
- [ ] Graph-API is running and healthy
- [ ] Learning-Graph is running and healthy
- [ ] Concept-Analytics is running and healthy
- [ ] Concept-Trainer and Growable variants are running
- [ ] Concept-Trainer-Embedder and Growable variants are running
- [ ] Concept-Analyzer is running and healthy

## 4. Ingestion/Event Services
- [ ] Crawler is running and healthy
- [ ] Event-Mux is running and healthy
- [ ] Batch-Embedder is running and healthy
- [ ] Output-Engine is running and healthy

## 5. Chat/LLM & Adapters
- [ ] Dual-Chat-Router is running and healthy
- [ ] Masterchat-Core is running and healthy
- [ ] Graph-Concept-Adapter is running and healthy

## 6. Supporting/Other Services
- [ ] Auditor is running and healthy
- [ ] Debugger is running and healthy
- [ ] Drift-Exporter is running and healthy
- [ ] Monitoring is running and healthy
- [ ] PDF-Trainer is running and healthy (if enabled)

## 7. UI (if enabled)
- [ ] UI service is running and accessible

## 8. General Health
- [ ] All containers show status 'Up' (not restarting/unhealthy)
- [ ] All health checks pass
- [ ] No critical errors in logs

## 9. Git (if Docker stack isn't working)
- [ ] Commit (or amend) your changes so that all files (Dockerfile(s), docker-compose.yml, and any other files that run your docker services) are included.
- [ ] Push your commit (or force-push if amended) so that your GitHub repository is updated with the latest (or a working) version of your Docker stack.

## 10. Miscellaneous
- [ ] Check logs (e.g. via docker logs or Grafana/Loki) for any errors or warnings.
- [ ] Verify that endpoints (e.g. dual-chat endpoint) are reachable and return a valid response (e.g. via curl or a test script).

---

## Troubleshooting Steps
- [ ] Check logs for any container that is restarting or unhealthy
- [ ] Check for dependency errors (e.g., Redis or Qdrant not available)
- [ ] Check for port conflicts or resource limits
- [ ] Check .env and configuration files for correct values
- [ ] Rebuild images if requirements or Dockerfiles have changed

---

_Use this checklist as you work through each service and dependency. Mark off each item as it becomes healthy/running._ 