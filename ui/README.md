# Lumina UI Setup Guide — Extended

## 0  Back‑end quick‑start (do this before running the UI)
```bash
# from repo root
docker compose up -d                # spins Redis, Graph‑API, Event‑Mux, MasterChat, Crawler…
./scripts/align_ports.sh            # writes ui/.env with correct host‑ports
./scripts/smoke.sh                  # optional – verifies /health endpoints
```
If you're on Windows: run these inside WSL so Docker sockets line up.

## 1  Prerequisites
| Tool  | Version         | Notes                        |
|-------|-----------------|------------------------------|
| Node  | 18 LTS or later | nvm install 18 && nvm use 18 |
| npm   | 9.x or later    | ships with Node 18           |
| Git   | any recent      |                              |

## 2  Clone & install
```bash
git clone <repo>
cd ui
npm install
```

## 3  Environment variables
ui/.env is generated automatically by scripts/align_ports.sh and looks like:
```env
VITE_API_URL=http://localhost:8201
VITE_EVENTS_URL=http://localhost:8101
VITE_MASTERCHAT_LOGS=http://localhost:8301/planner/logs
VITE_WS_URL=ws://localhost:8101/ws
```
If you prefer manual setup:
```bash
cp .env.example .env        # edit the port numbers if you changed compose
```

## 4  Run the dev server
```bash
npm run dev
```
Vite will print the exact URL (default http://localhost:5173).
If that port is taken, Vite picks the next one and prints it in the console.

## 5  Type‑checking & linting
```bash
npm run type-check          # tsc --noEmit
npm run lint                # ESLint + Prettier
```

## 6  Build & preview (prod mode)
```bash
npm run build               # outputs to dist/
npm run preview             # serves dist/ on :4173
```

## 7  Testing
### 7.1  Unit / component
```bash
npm run test            # vitest
npm run test:watch
npm run test:coverage
```
### 7.2  Cypress E2E
```bash
# interactive
npm run cypress:open

# headless (with video + snapshots)
npm run cypress:ci
```
**Spec shortcuts**

| Script                  | What it runs                |
|------------------------|-----------------------------|
| npm run cypress:smoke   | just happy_path             |
| npm run cypress:stubbed | wiki QA tests with stubbed API |
| npm run cypress:live    | live wiki crawl tests       |

Tip: the Cypress runner respects CYPRESS_BASE_URL.
CI sets this to the Vite dev URL printed by the backend‑up step.

## 8  Dockerising the UI (static export)
```bash
docker build -t lumina-ui .
docker run -p 80:80 lumina-ui
```
(Nginx image, serves dist/.)

## 9  Project layout
```csharp
ui/
├─ src/
│  ├─ components/       # Vue components
│  ├─ hooks/            # composables (GraphSocket, ErrorReporter)
│  ├─ styles/           # Tailwind + utilities.css
│  └─ router/
├─ cypress/             # E2E tests + snapshots
├─ public/              # static assets
└─ tests/               # vitest unit specs
```

## 10  Key runtime deps
| Library      | Role                        |
|--------------|-----------------------------|
| Vue 3        | UI framework (Composition API) |
| Vite 5       | Dev server / bundler        |
| Tailwind 3   | Utility CSS                 |
| Pinia 2      | Global state                |
| Chart.js 4   | MetricsPanel charts         |
| D3 v7        | FractalView & SubgraphView  |
| Socket.IO 4  | Real‑time WS/SSE            |
| Cypress 13   | E2E + visual regression     |

## 11  Dev guidelines
- TypeScript first – no new .js unless a lib requires it.
- Accessibility – use Testing‑Library queries (findByRole) in specs.
- State – keep cross‑component state in Pinia; local state in setup().
- Styling – Tailwind utilities; no global CSS except resets.
- Snapshots – when UI changes are intentional, run npx cypress run –u and commit new images.

## 12  Troubleshooting
| Symptom                        | Fix                                                                 |
|--------------------------------|---------------------------------------------------------------------|
| Blank graph                    | Check browser console. 404 on /hierarchy → backend not running or wrong VITE_API_URL. |
| WebSocket disconnect           | Verify VITE_WS_URL. Event‑Mux must expose /ws; firewall can't block port 8101. |
| Cypress fails "element not found" | Backend data seeded? Run the mini‑crawl via /tasks then reload spec. Increase defaultCommandTimeout if charts animate slowly. |
| Build fails on "out of memory" | NODE_OPTIONS=--max_old_space_size=4096 npm run build on small CI runners. |

## 13  Contributing flow
```bash
git checkout -b feat/my-feature

Code ➜ npm run type-check + unit tests

Add/modify Cypress spec if UI changes

git commit -m "feat: my feature"

Open PR, wait for green CI (lint + vitest + Cypress)

Reviewer merges to main
```

## 14  License
[MIT (placeholder) – update if different]

---

That's it — once the back-end stack is up and .env is aligned,
npm run dev should hot‑reload and display live graph/events.
If the Cypress run is still red, pull the artifact screenshots and adjust selectors or snapshots.

Happy hacking! 