# E2E Test Troubleshooting Guide

## Overview
This guide documents the failures encountered during the end-to-end modular test and provides steps to resolve them.

## Current System State (Updated)

### Healthy/Working Services
- ✅ **Redis**: Running, password-protected, healthy
- ✅ **Qdrant**: Healthy
- ✅ **Concept Dictionary**: Healthy
- ✅ **Output Engine**: Healthy, API key/authentication issues resolved
- ✅ **Ollama**: Healthy
- ✅ **Dead-Letter UI**: Healthy
- ✅ **Neo4j, Prometheus, Grafana, Alertmanager, Batch Embedder, Drift Exporter, Learning Graph, Concept Analyzer, Concept Analytics**: Healthy

### Unhealthy/Restarting Services
- ❌ **Feedback Logger**: Unhealthy
- ❌ **Ingest Gateway**: Restarting
- ❌ **Crawler**: Health: starting
- ❌ **Retrain Listener**: Restarting
- ❌ **RAG Coordinator**: Restarting
- ❌ **PDF Trainer**: Restarting
- ❌ **Graph Concept Adapter**: Restarting
- ❌ **Monitoring, Auditor, Debugger, Event Mux**: Unhealthy

## Recent Progress
- Redis security: Password and protected mode enabled, all services updated to use the new password.
- Output Engine: API key set, health check now passing, service is up.
- Dead-Letter UI: Port/configuration fixed, now healthy.

## What To Fix Next

### 1. Unhealthy/Restarting Services
- Check logs for each unhealthy/restarting service (e.g., `docker logs lumina-system-ingest-gateway-1 --tail 50`)
- Common issues:
  - Wrong Redis password in environment
  - Missing/incorrect dependencies (e.g., waiting for another service to be healthy)
  - Port conflicts
  - Resource limits (memory/CPU)
  - Bad/missing environment variables

### 2. Bridge/Functional Test Failures
- Output Engine → Redis:
  - If you see 401 Unauthorized, double-check API key usage in both the Output Engine and the test script.
  - If key not found in Redis, check Output Engine logs for errors writing to Redis.
- Feedback Logger → Redis:
  - 500 errors often mean Redis connection issues or bad/missing environment variables.
- Output Engine → Qdrant:
  - 404/401 errors: check Qdrant collection existence and API key usage.

### 3. Authentication Issues
- API Key Consistency:
  - Make sure all services and test scripts use the same API key (e.g., `changeme` or your custom key).
  - Set `LUMINA_API_KEY` in your environment and Docker Compose for all relevant services.

### 4. Service Startup Order
- Some services depend on others being healthy.
  - Start core services (Redis, Qdrant, Neo4j) first.
  - Then start supporting services (Concept Dictionary, Trainer, Batch Embedder).
  - Then application services (Output Engine, Feedback Logger, Dead-Letter UI).

### 5. Health Check Endpoints
- If a service is healthy but health check fails, check the health endpoint implementation and what dependencies it checks.

## Recommended Next Steps

1. Check logs for all unhealthy/restarting services
   - Example:
     ```
     docker logs lumina-system-ingest-gateway-1 --tail 50
     docker logs lumina-system-feedback-logger-1 --tail 50
     ```
2. Verify all services use the correct Redis password and API key
   - Check `docker-compose.yml` and environment variables.
3. Restart services after fixing configs
   - Example:
     ```
     docker compose restart ingest-gateway feedback-logger ...
     ```
4. Re-run health checks and bridge tests
   - Use your `e2e_modular_test.py` script.
5. If a service still fails, share its logs for targeted debugging.

## Summary Table

| Service                | Status      | Fix Next? | Notes                                  |
|------------------------|-------------|-----------|----------------------------------------|
| Redis                  | Healthy     | No        | Password protected, all good           |
| Qdrant                 | Healthy     | No        |                                        |
| Output Engine          | Healthy     | No        | API key set, health check passing      |
| Concept Dictionary     | Healthy     | No        |                                        |
| Feedback Logger        | Unhealthy   | Yes       | Check Redis config, logs               |
| Ingest Gateway         | Restarting  | Yes       | Check logs, Redis config               |
| Crawler                | Starting    | Yes       | Check logs, dependencies               |
| Retrain Listener       | Restarting  | Yes       | Check logs, dependencies               |
| RAG Coordinator        | Restarting  | Yes       | Check logs, dependencies               |
| PDF Trainer            | Restarting  | Yes       | Check logs, dependencies               |
| Graph Concept Adapter  | Restarting  | Yes       | Check logs, dependencies               |
| Monitoring/Auditor/etc | Unhealthy   | Yes       | Check logs, Redis config               |

---

**Continue fixing services in the order above, focusing on logs and configuration for each.**

## Test Results Summary

### Failed Health Checks
1. **Redis**
   - Error: Connection aborted, Remote end closed connection without response
   - Status: Critical (affects multiple services)
   - Port: 6379

2. **Concept Dictionary**
   - Error: Connection aborted, Remote end closed connection without response
   - Status: Critical (affects concept-related functionality)
   - Port: 8828

3. **Concept Trainer Growable**
   - Error: Connection refused
   - Status: High
   - Port: 8905

4. **Batch Embedder**
   - Error: Connection refused
   - Status: High
   - Port: 8709

### Working Services
✅ Event Mux (Port: 8817)
✅ Monitoring (Port: 8824)
✅ Auditor (Port: 8811)
✅ Debugger (Port: 8814)
✅ Drift Exporter (Port: 8816)
✅ Neo4j (Port: 7474)
✅ Qdrant (Port: 6333)
✅ Ollama (Port: 11434)
✅ Prometheus (Port: 9090)
✅ Grafana (Port: 3000)
✅ Alertmanager (Port: 9093)
✅ Dead-Letter UI (Port: 8602)

### Failed Bridge/Functional Tests
1. **Output Engine → Redis**
   - Error: 401 Unauthorized
   - Endpoint: http://localhost:9000/output
   - Status: Critical (affects core functionality)

2. **Feedback Logger → Redis**
   - Error: Connection refused
   - Endpoint: http://localhost:8900/feedback
   - Status: High

3. **Output Engine → Qdrant**
   - Error: 401 Unauthorized
   - Endpoint: http://localhost:9000/output
   - Status: Critical

4. **Concept Dictionary**
   - Error: Connection aborted
   - Endpoint: http://localhost:8828/concepts
   - Status: High

5. **Dead-Letter Flow**
   - Error: Connection refused
   - Endpoint: http://localhost:8601/submit/file
   - Status: Medium

## Troubleshooting Steps

### 1. Authentication Issues (401 Unauthorized)
- [ ] Check Output Engine authentication configuration
- [ ] Verify API keys or tokens are properly set
- [ ] Review authentication middleware logs
- [ ] Check if authentication service is running

### 2. Connection Refused Issues
For each service showing "connection refused":
- [ ] Verify service is running (`docker ps` or service status)
- [ ] Check if port is actually in use (`netstat -ano | findstr <port>`)
- [ ] Verify firewall settings
- [ ] Check service logs for startup errors

### 3. Connection Aborted Issues
For Redis and Concept Dictionary:
- [ ] Check service logs for connection termination reasons
- [ ] Verify memory usage and resource limits
- [ ] Check for any timeout settings
- [ ] Review network stability

### 4. Service Startup Order
1. Start core services first:
   - [ ] Redis
   - [ ] Neo4j
   - [ ] Qdrant

2. Start supporting services:
   - [ ] Concept Dictionary
   - [ ] Concept Trainer
   - [ ] Batch Embedder

3. Start monitoring services:
   - [ ] Prometheus
   - [ ] Grafana
   - [ ] Alertmanager

4. Start application services:
   - [ ] Output Engine
   - [ ] Feedback Logger
   - [ ] Dead-Letter services

## Verification Steps
After fixing each issue:
1. Run individual health checks
2. Test bridge connections
3. Run full E2E test

## Common Commands
```bash
# Check service status
docker ps

# Check port usage
netstat -ano | findstr <port>

# Check service logs
docker logs <service_name>

# Test service connectivity
curl http://localhost:<port>/health
```

## Notes
- Keep track of any configuration changes made
- Document any workarounds implemented
- Note any services that require specific startup order
- Record any environment-specific issues

## Investigation Results

### Redis Issues
- Redis container is running and marked as healthy
- Port 6379 is active and listening
- **Critical Finding**: Redis logs show security attack detection:
  ```
  Possible SECURITY ATTACK detected. It looks like somebody is sending POST or Host: commands to Redis.
  ```
- This is causing connections to be aborted
- **Action Items**:
  1. [ ] Review Redis security configuration
  2. [ ] Check if any services are sending malformed requests
  3. [ ] Verify Redis bind settings and network access
  4. [ ] Consider updating Redis security policies

### Output Engine Issues
- Container is running but marked as unhealthy
- All requests returning 401 Unauthorized
- **Action Items**:
  1. [ ] Check Output Engine environment variables for auth tokens
  2. [ ] Verify authentication middleware configuration
  3. [ ] Review API key or token generation process
  4. [ ] Check if auth service is properly integrated

## Immediate Next Steps
1. [ ] **High Priority**: Address Redis security issue
   - Check Redis configuration file
   - Review network access patterns
   - Update security policies if needed

2. [ ] **High Priority**: Fix Output Engine authentication
   - Locate authentication configuration
   - Verify token generation and validation
   - Test with valid authentication

3. [ ] **Medium Priority**: After fixing above issues
   - Restart affected services
   - Verify Redis connectivity
   - Test Output Engine endpoints with proper auth

## Updated Service Status
- Redis: Running but with security issues
- Output Engine: Running but authentication failing
- Other services: Status unchanged

## Next Steps
1. [ ] Address authentication issues first
2. [ ] Restart failed services in correct order
3. [ ] Verify each service individually
4. [ ] Run bridge tests
5. [ ] Execute full E2E test

## Redis Security Implementation Status

### Changes Made
1. ✅ Created backup of Redis data
2. ✅ Set Redis password to "02211998"
3. ✅ Enabled protected mode
4. ✅ Verified new settings are active
5. ✅ Updated Redis configurations in services:
   - Batch Embedder
   - Concept Analyzer
   - Concept Dictionary
   - Crawler
   - Concept Trainer
   - Concept Trainer Growable
   - Ingest Gateway

### Current Configuration
```redis
requirepass 02211998
protected-mode yes
```

### Next Steps
1. [ ] Restart all updated services to apply new Redis configuration
   ```bash
   # Example restart command (adjust based on your setup)
   docker-compose restart batch-embedder concept-analyzer concept-dictionary crawler concept-trainer concept-trainer-growable ingest-gateway
   ```

2. [ ] Verify Redis connectivity
   - Check service logs for successful connections
   - Monitor for any authentication errors
   - Verify all services can communicate with Redis

3. [ ] Run E2E tests again
   - This will verify all services work with new Redis settings
   - Pay attention to any Redis-related failures
   - Check for any new connection issues

### Service Status
- Redis: Secured with password and protected mode
- Services: Updated with new Redis configuration
- Next Action: Restart services to apply changes

### Service Update Instructions
For each service using Redis, update the connection string to include the password:
```python
# Example Redis connection update
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    password='02211998',
    decode_responses=True
)
```

## Latest Test Results (Updated)

### Current Service Status (As of Latest Test)

#### Working Services
✅ Redis (Port 6379) - Basic connectivity restored
✅ Concept Dictionary (Port 8828) - Health check passing
✅ Concept Trainer Growable (Port 8710) - Health check passing
✅ Batch Embedder (Port 8709) - Health check passing
✅ Event Mux (Port 8817) - Health check passing
✅ Monitoring (Port 8824) - Health check passing
✅ Auditor (Port 8811) - Health check passing
✅ Debugger (Port 8814) - Health check passing
✅ Drift Exporter (Port 8816) - Health check passing
✅ Neo4j (Port 7474) - Health check passing
✅ Qdrant (Port 6333) - Health check passing
✅ Ollama (Port 11434) - Health check passing
✅ Prometheus (Port 9090) - Health check passing
✅ Grafana (Port 3000) - Health check passing
✅ Alertmanager (Port 9093) - Health check passing

#### Failed Services
❌ Dead-Letter UI (Port 8602) - Connection refused
❌ Output Engine (Port 9000) - 401 Unauthorized
❌ Feedback Logger (Port 8900) - 500 Internal Server Error

#### Failed Bridge Tests
❌ Output Engine → Redis bridge - Key not found
❌ Feedback Logger → Redis bridge - 500 Internal Server Error
❌ Output Engine → Qdrant bridge - 404 Not Found
❌ Dead-Letter UI health check - Connection refused
❌ Dead-Letter flow test - Connection aborted

## Progress Made
1. ✅ Redis connectivity restored
2. ✅ Concept Dictionary service operational
3. ✅ Concept Trainer Growable port corrected (8710)
4. ✅ Most core services now passing health checks

## Current Issues

### 1. Authentication Issues
- Output Engine returning 401 Unauthorized
- API key validation may be failing
- Action Items:
  - [ ] Verify LUMINA_API_KEY environment variable
  - [ ] Check Output Engine authentication middleware
  - [ ] Review API key validation logic

### 2. Service Connection Issues
- Dead-Letter UI not accessible on port 8602
- Feedback Logger returning 500 errors
- Action Items:
  - [ ] Check Dead-Letter UI container status
  - [ ] Review Feedback Logger error logs
  - [ ] Verify service dependencies

### 3. Bridge Test Failures
- Output Engine to Redis: Key not found
- Output Engine to Qdrant: 404 Not Found
- Action Items:
  - [ ] Verify Redis key storage mechanism
  - [ ] Check Qdrant collection existence
  - [ ] Review bridge test implementation

## Next Steps

### High Priority
1. [ ] Fix Output Engine authentication
   ```bash
   # Verify API key
   echo $LUMINA_API_KEY
   # Check Output Engine logs
   docker logs lumina-system-output-engine-1
   ```

2. [ ] Investigate Dead-Letter UI
   ```bash
   # Check container status
   docker ps | grep dead-letter-ui
   # Review logs
   docker logs lumina-system-dead-letter-ui-1
   ```

3. [ ] Debug Feedback Logger 500 errors
   ```bash
   # Check logs
   docker logs lumina-system-feedback-logger-1
   # Verify Redis connection
   docker exec -it lumina-system-redis-1 redis-cli -a 02211998 ping
   ```

### Medium Priority
1. [ ] Verify Redis key storage
2. [ ] Check Qdrant collection setup
3. [ ] Review bridge test implementations

## Service Dependencies
1. Core Services:
   - Redis (Port 6379)
   - Neo4j (Port 7474)
   - Qdrant (Port 6333)

2. Supporting Services:
   - Concept Dictionary (Port 8828)
   - Concept Trainer Growable (Port 8710)
   - Batch Embedder (Port 8709)

3. Application Services:
   - Output Engine (Port 9000)
   - Feedback Logger (Port 8900)
   - Dead-Letter UI (Ports 8601, 8602)

## Verification Steps
After implementing fixes:
1. [ ] Run individual health checks
2. [ ] Test bridge connections
3. [ ] Execute full E2E test

## Common Commands
```bash
# Check service status
docker ps

# Check service logs
docker logs <service_name>

# Test service health
curl -H "X-API-Key: $LUMINA_API_KEY" http://localhost:<port>/health

# Check Redis connectivity
docker exec -it lumina-system-redis-1 redis-cli -a 02211998 ping
```

## Notes
- Redis password is set to "02211998"
- API key should be set in environment as LUMINA_API_KEY
- Most services now passing health checks
- Focus on authentication and bridge test issues 