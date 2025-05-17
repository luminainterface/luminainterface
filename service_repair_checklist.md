# Service Repair Checklist

This document tracks all known issues from the latest modular system test and their possible fixes. Check off each item as it is resolved.

---

## Checklist

- [ ] **Redis HTTP health check failed**
  - Ignore or remove HTTP check; use direct Redis ping instead.

- [ ] **Concept Trainer Growable not running**
  - Start service; check logs; resolve port conflicts.

- [ ] **Batch Embedder not running**
  - Start service; check logs; resolve port conflicts.

- [ ] **Dead-Letter UI not running**
  - Start service; check logs; resolve port conflicts.

- [ ] **Output Engine → Redis bridge test failed (401 Unauthorized)**
  - Add API key to test script; verify API key configuration.

- [ ] **Feedback Logger not running**
  - Start service; check logs; resolve port conflicts.

- [ ] **Output Engine → Qdrant bridge test failed (401 Unauthorized)**
  - Add API key to test script; verify API key configuration.

- [ ] **Concept Dictionary /concepts GET test failed (401 Unauthorized)**
  - Add API key to test script; verify API key configuration.

- [ ] **Dead-Letter UI health and flow tests failed**
  - Start service; check logs; resolve port conflicts.

- [ ] **Multiple services not running**
  - Use `docker compose ps`/`up -d`; check logs for failures.

- [ ] **API key not included in test script**
  - Update test script to include `X-API-Key` header.

- [ ] **DLQ submission test failed**
  - Start Dead-Letter UI/backend; check for port conflicts/misconfigurations.

---

## Progress Log

- Add notes here as you work through each item. 