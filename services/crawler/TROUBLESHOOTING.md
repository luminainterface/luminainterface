# Crawler Troubleshooting Guide

This guide helps you diagnose and fix common issues with the crawler service in this project.

---

## 1. Uvicorn Not Found / App Not Starting
**Symptom:**
- `exec: "uvicorn": executable file not found in $PATH: unknown`

**Fix:**
- Ensure `uvicorn` is in `services/crawler/requirements.txt`.
- Ensure the Dockerfile installs requirements:
  ```Dockerfile
  RUN pip install --no-cache-dir -r requirements.txt
  ```
- Force a clean build:
  ```sh
  docker compose build --no-cache crawler
  docker compose up -d crawler
  ```
- Check build logs for `Successfully installed uvicorn`.

---

## 2. Health Endpoint 404
**Symptom:**
- Docker healthcheck fails with 404.

**Fix:**
- Ensure `/health` endpoint exists in `services/crawler/app/main.py`:
  ```python
  @app.get("/health")
  async def health():
      return {"status": "ok"}
  ```
- Rebuild and restart the crawler.
- Test with:
  ```sh
  curl http://localhost:8830/health
  ```

---

## 3. Connection Refused When Crawling
**Symptom:**
- `[Errno 111] Connection refused` in logs.

**Fix:**
- Check which service is being connected to (Wikipedia, Neo4j, Qdrant, etc.).
- Ensure the target service is running and healthy (`docker compose ps`).
- Ensure the crawler uses the correct host/port in its environment variables.
- Check Docker Compose networking (see below).

---

## 4. Docker Compose Networking Issues
**Symptom:**
- Services cannot resolve each other by name (e.g., `redis`).

**Fix:**
- Ensure all services are on the same network in `docker-compose.yml`:
  ```yaml
  networks:
    - lumina_net
  ```
- Use service names as hostnames (e.g., `redis:6379`).
- Restart with:
  ```sh
  docker compose down -v
  docker compose up -d redis crawler
  ```
- Test DNS resolution:
  ```sh
  docker compose exec crawler getent hosts redis
  ```

---

## 5. Build/Context Issues
**Symptom:**
- Code or requirements changes are not reflected in the container.

**Fix:**
- Ensure you are building from the correct context (`services/crawler`).
- Use `docker compose build --no-cache crawler` after changes.

---

## 6. Crawler Not Processing Queue
**Symptom:**
- Queue is filled, but no crawls are processed.

**Fix:**
- Check logs for `Processing inquiries` and `Starting crawl from ...`.
- Ensure the FastAPI app is running and the background task is started.
- Check for unhandled exceptions in the crawl loop.

---

## 7. Service Startup Order/Timing
**Symptom:**
- Crawler tries to connect to a service before it is ready.

**Fix:**
- Use `depends_on` in `docker-compose.yml`.
- Add healthchecks to all services.
- Add retry logic in the crawler for dependent services.

---

## 8. Environment Variable Issues
**Symptom:**
- Services use wrong host/port for dependencies.

**Fix:**
- Double-check all environment variables in `docker-compose.yml`.
- Use service names as hostnames.

---

## 9. Application Code Bugs
**Symptom:**
- Unexpected exceptions, missing logs, or logic errors.

**Fix:**
- Add more detailed logging and error handling.
- Check for unhandled exceptions in async tasks.

---

## 10. Resource Constraints
**Symptom:**
- Services crash, hang, or are killed.

**Fix:**
- Check Docker resource limits (CPU, memory, disk).
- Monitor container logs for OOM or disk errors.

---

**If you are stuck, try:**
- Forcing a clean build and restart.
- Checking logs for errors and stack traces.
- Verifying all services are healthy and on the same network.
- Asking for help with the exact error message and recent logs. 