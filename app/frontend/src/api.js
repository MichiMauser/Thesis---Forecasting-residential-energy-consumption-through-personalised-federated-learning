// /static către :8000; în build FastAPI le servește pe amândouă, deci căile

async function jget(path) {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${path} -> ${r.status}`)
  return r.json()
}

async function jpost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : "{}",
  })
  if (!r.ok) throw new Error(`${path} -> ${r.status}`)
  return r.json()
}

export const api = {
  health:        () => jget("/api/health"),
  runs:          () => jget("/api/runs"),
  runRounds:     (id) => jget(`/api/runs/${id}/rounds`),
  runImages:     (id) => jget(`/api/runs/${id}/images`),
  latestRounds:  () => jget("/api/rounds/latest"),
  clients:       () => jget("/api/clients"),
  clientMetrics: (id) => jget(`/api/clients/${id}/metrics`),
  clientImages:  (id) => jget(`/api/clients/${id}/images`),
  comparison:     () => jget("/api/comparison"),
  comparisonLive: () => jget("/api/comparison-live"),
  eda:            () => jget("/api/eda"),

  flStatus: () => jget("/api/fl/status"),
  flUp:     () => jpost("/api/fl/up"),
  flDown:   () => jpost("/api/fl/down"),
  flRun:    (cfg) => jpost("/api/fl/run", cfg),

  availableClients: () => jget("/api/available-clients"),
  getFederation:    () => jget("/api/federation/clients"),
  setFederation:    (clients) => jpost("/api/federation/clients", { clients }),

  // SSE: întoarce un EventSource; apelantul își leagă onmessage / addEventListener
  stream: (runId) =>
    new EventSource(runId ? `/api/stream?run_id=${runId}` : "/api/stream"),

  // URL-ul unei imagini din outputs/, servită la /static
  staticUrl: (rel) => (rel ? `/static/${rel}` : null),
}
