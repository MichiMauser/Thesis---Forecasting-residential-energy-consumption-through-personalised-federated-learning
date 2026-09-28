<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue"
import { api } from "../api.js"

const rows = ref([])
const selection = ref([])
const err = ref("")
const metric = ref("R2")
const edaImages = ref([])
let timer = null

// comparație live pentru selecția curentă: persistența apare automat, iar
// metodele FL pe măsură ce sunt rulate. Se afișează doar cele existente.
const METHOD_ORDER = ["Persistence", "FedAvg", "FedProx", "PerFedAvg", "PerFedAvg-Personal"]
const COLORS = {
  Persistence: "#888", FedAvg: "#DD8452", FedProx: "#55A868",
  PerFedAvg: "#4C72B0", "PerFedAvg-Personal": "#8172B2",
}
const METRICS = ["R2", "MAE", "RMSE", "Spike_MAE", "MAPE"]

async function refresh() {
  try {
    const r = await api.comparisonLive()
    rows.value = r.rows || []
    selection.value = r.clients || []
    err.value = ""
  } catch (e) { err.value = String(e) }
  try {
    edaImages.value = (await api.eda()).images || []
  } catch { /* păstrăm ultima valoare */ }
}

onMounted(() => { refresh(); timer = setInterval(refresh, 4000) })
onUnmounted(() => clearInterval(timer))

const households = computed(() =>
  [...new Set(rows.value.map(r => r.household))].sort())
const methods = computed(() =>
  METHOD_ORDER.filter(m => rows.value.some(r => r.method === m)))

function val(h, m, metricName) {
  const r = rows.value.find(x => x.household === h && x.method === m)
  return r ? r[metricName] : null
}

const barOption = computed(() => ({
  tooltip: { trigger: "axis" },
  legend: { textStyle: { color: "#9aa6bd" }, top: 0, type: "scroll" },
  grid: { left: 48, right: 16, top: 40, bottom: 60 },
  xAxis: { type: "category", data: households.value,
           axisLabel: { color: "#9aa6bd", rotate: 20 } },
  yAxis: { type: "value", name: metric.value, axisLabel: { color: "#9aa6bd" } },
  series: methods.value.map(m => ({
    name: m, type: "bar",
    data: households.value.map(h => val(h, m, metric.value)),
    itemStyle: { color: COLORS[m] || "#666" },
  })),
}))

const avg = computed(() => {
  const out = []
  for (const m of methods.value) {
    const vs = households.value.map(h => val(h, m, "R2")).filter(v => v != null)
    const ms = households.value.map(h => val(h, m, "MAE")).filter(v => v != null)
    out.push({
      method: m,
      R2: vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null,
      MAE: ms.length ? ms.reduce((a, b) => a + b, 0) / ms.length : null,
    })
  }
  return out.sort((a, b) => (b.R2 ?? -9) - (a.R2 ?? -9))
})
</script>

<template>
  <h2>Comparison — current selection ({{ selection.length }} households)</h2>
  <p v-if="err" class="muted">{{ err }}</p>

  <div class="panel">
    <h3 style="margin-top:0">Exploratory data analysis — selected households
      <span class="muted" style="font-size:13px">— consumption distribution &amp; average load profile, for the currently selected clients</span>
    </h3>
    <div v-for="img in edaImages" :key="img.name" style="margin-bottom:10px">
      <div class="muted" style="font-size:12px; margin-bottom:4px">{{ img.title }}</div>
      <a :href="api.staticUrl(img.url)" target="_blank">
        <img class="artefact" :src="api.staticUrl(img.url)" :alt="img.title" />
      </a>
    </div>
    <p v-if="!edaImages.length" class="muted">Generating comparison for the selected clients… (appears in a few seconds)</p>
  </div>

  <div class="panel">
    <h3 style="margin-top:0">Method comparison — test metrics
      <span class="muted" style="font-size:13px">— Persistence auto; FL methods fill in as you run them</span>
    </h3>
    <div style="margin-bottom:10px">
      <label>metric</label>
      <select v-model="metric" style="background:var(--panel-2);color:var(--text);border:1px solid var(--border);padding:6px;border-radius:6px">
        <option v-for="m in METRICS" :key="m" :value="m">{{ m }}</option>
      </select>
    </div>
    <VChart v-if="households.length && methods.length" class="chart tall" :option="barOption" autoresize />
    <p v-else class="muted">
      No runs yet for this selection. Persistence computes automatically; run a
      method from <strong>Overview → Trigger run</strong> (FedAvg / FedProx /
      PerFedAvg) and it appears here.
    </p>
  </div>

  <div class="panel" v-if="methods.length">
    <h3 style="margin-top:0">Global average (mean across households)</h3>
    <table>
      <thead><tr><th>Method</th><th>R2 ↑</th><th>MAE ↓</th></tr></thead>
      <tbody>
        <tr v-for="a in avg" :key="a.method">
          <td>{{ a.method }}</td>
          <td>{{ a.R2 != null ? a.R2.toFixed(4) : "—" }}</td>
          <td>{{ a.MAE != null ? a.MAE.toFixed(4) : "—" }}</td>
        </tr>
      </tbody>
    </table>
    <p class="muted" style="font-size:12px">
      Metrics are the final-round <strong>test</strong> results of the latest run of
      each method, for the currently selected households. PerFedAvg-Personal is the
      per-client fine-tuned model.
    </p>
  </div>
</template>
