<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from "vue"
import { useRoute } from "vue-router"
import { api } from "../api.js"

const route = useRoute()
const records = ref([])          // all streamed records for the followed run
const runId = ref(route.query.run_id || null)
const connected = ref(false)
const images = ref([])           // per-run prediction PNGs (loaded on completion)
let es = null

function connect() {
  es = api.stream(runId.value)
  es.addEventListener("run", (e) => { runId.value = JSON.parse(e.data).run_id })
  es.onopen = () => { connected.value = true }
  es.onerror = () => { connected.value = false }
  es.onmessage = (e) => {
    try {
      const rec = JSON.parse(e.data)
      if (rec && rec.scope) records.value.push(rec)
    } catch { /* mesaj keepalive sau non-JSON */ }
  }
}

onMounted(connect)
onUnmounted(() => es && es.close())

const globalRows = computed(() =>
  records.value.filter(r => r.scope === "global").sort((a, b) => a.round - b.round))

const clientRows = computed(() => records.value.filter(r => r.scope === "client"))

const method = computed(() => records.value.find(r => r.method)?.method || null)

const runComplete = computed(() => clientRows.value.some(r => r.split === "test"))

watch([runComplete, runId], async ([done, rid]) => {
  if (done && rid && !images.value.length) {
    try { images.value = (await api.runImages(rid)).clients || [] }
    catch { }
  }
})

const latestRound = computed(() => {
  const rs = clientRows.value.map(r => r.round)
  return rs.length ? Math.max(...rs) : null
})

const latestClients = computed(() => {
  if (latestRound.value == null) return []
  const seen = {}
  for (const r of clientRows.value) if (r.round === latestRound.value) seen[r.lclid] = r
  return Object.values(seen).sort((a, b) => a.lclid.localeCompare(b.lclid))
})

// istoricul complet per client: toate rundele, cea mai nouă prima
const showAllRounds = ref(false)
const roundsDesc = computed(() => {
  const byRound = {}
  for (const r of clientRows.value) (byRound[r.round] ??= []).push(r)
  return Object.keys(byRound)
    .map(Number)
    .sort((a, b) => b - a)
    .map(rnd => ({
      round: rnd,
      clients: byRound[rnd].slice().sort((a, b) => a.lclid.localeCompare(b.lclid)),
    }))
})

const graphOption = computed(() => {
  const clients = latestClients.value
  const R = 320, n = clients.length || 1
  const nodes = [{
    name: "Server", x: 0, y: 0, symbolSize: 54,
    itemStyle: { color: "#DD8452" }, label: { show: true },
  }]
  const links = []
  const maxDrift = Math.max(0.001, ...clients.map(c => c.drift || 0))
  clients.forEach((c, i) => {
    const ang = (2 * Math.PI * i) / n - Math.PI / 2
    nodes.push({
      name: c.lclid, x: R * Math.cos(ang), y: R * Math.sin(ang),
      symbolSize: 30 + 26 * ((c.drift || 0) / maxDrift),
      itemStyle: { color: (c.R2 ?? 0) >= 0.5 ? "#55A868" : "#C44E52" },
      value: c.R2,
    })
    links.push({
      source: "Server", target: c.lclid,
      lineStyle: { width: 1 + 6 * ((c.drift || 0) / maxDrift) },
      label: { show: true, formatter: () => (c.drift != null ? c.drift.toFixed(2) : ""), color: "#9aa6bd" },
    })
  })
  return {
    tooltip: {
      formatter: (p) => p.dataType === "node" && p.data.value != null
        ? `${p.name}<br/>R2 = ${Number(p.data.value).toFixed(3)}` : p.name,
    },
    series: [{
      type: "graph", layout: "none", roam: true,
      label: { show: true, color: "#e6eaf2", fontSize: 11 },
      edgeLabel: { fontSize: 10 },
      data: nodes, links,
      lineStyle: { color: "#3a4252", curveness: 0 },
    }],
  }
})

const convOption = computed(() => {
  const rounds = globalRows.value.map(r => r.round)
  return {
    tooltip: { trigger: "axis" },
    legend: { textStyle: { color: "#9aa6bd" }, top: 0 },
    grid: { left: 48, right: 48, top: 34, bottom: 32 },
    xAxis: { type: "category", data: rounds, name: "round",
             axisLabel: { color: "#9aa6bd" } },
    yAxis: [
      { type: "value", name: "R2", axisLabel: { color: "#9aa6bd" } },
      { type: "value", name: "MAE", axisLabel: { color: "#9aa6bd" } },
    ],
    series: [
      { name: "R2", type: "line", smooth: true, data: globalRows.value.map(r => r.R2),
        itemStyle: { color: "#4C72B0" } },
      { name: "MAE", type: "line", smooth: true, yAxisIndex: 1,
        data: globalRows.value.map(r => r.MAE), itemStyle: { color: "#DD8452" } },
    ],
  }
})
</script>

<template>
  <h2>Live Round
    <span class="muted" style="font-size:14px; font-weight:400">
      — run {{ runId || "(latest)" }}
      <span v-if="method" class="pill">{{ method }}</span>
      <span class="dot" :class="connected ? 'on' : 'off'"></span>
      {{ connected ? "streaming" : "disconnected" }}
    </span>
  </h2>

  <div class="row">
    <div class="panel" style="flex:2">
      <h3 style="margin-top:0">Communication graph
        <span class="muted" style="font-size:13px">— round {{ latestRound ?? "—" }} (edge width = weight drift)</span>
      </h3>
      <VChart v-if="latestClients.length" class="chart tall" :option="graphOption" autoresize />
      <p v-else class="muted">Waiting for the first round… trigger a run from Overview.</p>
    </div>
    <div class="panel" style="flex:1">
      <h3 style="margin-top:0">Global convergence</h3>
      <VChart v-if="globalRows.length" class="chart tall" :option="convOption" autoresize />
      <p v-else class="muted">No global rounds yet.</p>
    </div>
  </div>

  <div class="panel">
    <h3 style="margin-top:0">Per-client — last round ({{ latestRound ?? "—" }})</h3>
    <table v-if="latestClients.length">
      <thead>
        <tr><th>Client</th><th>R2</th><th>MAE</th><th>RMSE</th><th>Spike_MAE</th><th>drift</th><th>drift_rel</th><th>split</th></tr>
      </thead>
      <tbody>
        <tr v-for="c in latestClients" :key="c.lclid">
          <td>{{ c.lclid }}</td>
          <td>{{ c.R2?.toFixed(4) }}</td>
          <td>{{ c.MAE?.toFixed(4) }}</td>
          <td>{{ c.RMSE?.toFixed(4) }}</td>
          <td>{{ c.Spike_MAE?.toFixed(4) }}</td>
          <td>{{ c.drift?.toFixed(3) }}</td>
          <td>{{ c.drift_rel?.toFixed(3) }}</td>
          <td>{{ c.split }}</td>
        </tr>
      </tbody>
    </table>
    <p v-else class="muted">No client records yet.</p>
    <button v-if="clientRows.length" class="ghost" style="margin-top:12px"
            @click="showAllRounds = true">
      View all rounds ({{ roundsDesc.length }})
    </button>
  </div>

  <div v-if="showAllRounds" class="modal-overlay" @click.self="showAllRounds = false">
    <div class="modal">
      <div class="modal-head">
        <h3 style="margin:0">All rounds — per-client history
          <span class="muted" style="font-size:13px">
            — {{ roundsDesc.length }} round(s), {{ clientRows.length }} records
          </span>
        </h3>
        <button class="ghost" @click="showAllRounds = false">Close ✕</button>
      </div>
      <div class="modal-body">
        <div v-for="grp in roundsDesc" :key="grp.round" style="margin-bottom:18px">
          <h4 style="margin:6px 0">Round {{ grp.round }}
            <span v-if="grp.round === latestRound" class="pill" style="text-transform:none">latest</span>
          </h4>
          <table>
            <thead>
              <tr><th>Client</th><th>R2</th><th>MAE</th><th>RMSE</th><th>Spike_MAE</th><th>drift</th><th>drift_rel</th><th>split</th></tr>
            </thead>
            <tbody>
              <tr v-for="c in grp.clients" :key="c.lclid">
                <td>{{ c.lclid }}</td>
                <td>{{ c.R2?.toFixed(4) }}</td>
                <td>{{ c.MAE?.toFixed(4) }}</td>
                <td>{{ c.RMSE?.toFixed(4) }}</td>
                <td>{{ c.Spike_MAE?.toFixed(4) }}</td>
                <td>{{ c.drift?.toFixed(3) }}</td>
                <td>{{ c.drift_rel?.toFixed(3) }}</td>
                <td>{{ c.split }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <div class="panel" v-if="runComplete">
    <h3 style="margin-top:0">Test predictions — this run
      <span class="muted" style="font-size:13px">
        — global model{{ method === 'perfedavg' ? ' + personalised' : '' }} per client
      </span>
    </h3>
    <p v-if="!images.length" class="muted">Rendering prediction PNGs…</p>
    <div v-for="c in images" :key="c.lclid" style="margin-bottom:18px">
      <h4 style="margin:6px 0">{{ c.lclid }}</h4>
      <div class="row" style="flex-wrap:wrap">
        <a v-if="c.predictions" :href="api.staticUrl(c.predictions)" target="_blank">
          <img :src="api.staticUrl(c.predictions)" alt="global predictions"
               style="max-width:520px; width:100%; border-radius:6px" />
        </a>
        <a v-if="c.personalised" :href="api.staticUrl(c.personalised)" target="_blank">
          <img :src="api.staticUrl(c.personalised)" alt="personalised predictions"
               style="max-width:520px; width:100%; border-radius:6px" />
        </a>
      </div>
    </div>
  </div>
</template>
