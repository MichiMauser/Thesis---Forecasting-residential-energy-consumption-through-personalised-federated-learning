<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from "vue"
import { useRoute } from "vue-router"
import { api } from "../api.js"

const route = useRoute()
const lclid = computed(() => route.params.lclid)
const metrics = ref([])
const images = ref({})
const err = ref("")

const METHOD_ORDER = ["Persistence", "Local", "FedAvg", "FedProx",
  "PerFedAvg-Global", "PerFedAvg-Personal", "Centralized-Global"]

// EDA se generează în fundal la prima accesare; reîncărcăm până apare
let pollTimer = null
const pollLeft = ref(0)
const POLL_MAX = 12        // ~36s la 3s/încercare
const POLL_MS = 3000

function stopPoll() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null }
}

function schedulePoll() {
  stopPoll()
  if (images.value.eda || pollLeft.value <= 0) return
  pollTimer = setTimeout(async () => {
    pollLeft.value -= 1
    try { images.value = await api.clientImages(lclid.value) } catch { /* ignor */ }
    schedulePoll()
  }, POLL_MS)
}

async function load() {
  err.value = ""
  stopPoll()
  pollLeft.value = POLL_MAX
  try {
    metrics.value = await api.clientMetrics(lclid.value)
    images.value = await api.clientImages(lclid.value)
  } catch (e) { err.value = String(e) }
  schedulePoll()
}
onMounted(load)
watch(lclid, load)
onUnmounted(stopPoll)

const ordered = computed(() =>
  [...metrics.value].sort((a, b) => METHOD_ORDER.indexOf(a.method) - METHOD_ORDER.indexOf(b.method)))

const r2Option = computed(() => ({
  tooltip: { trigger: "axis" },
  grid: { left: 48, right: 16, top: 16, bottom: 70 },
  xAxis: { type: "category", data: ordered.value.map(m => m.method),
           axisLabel: { color: "#9aa6bd", rotate: 28 } },
  yAxis: { type: "value", name: "R2", axisLabel: { color: "#9aa6bd" } },
  series: [{ type: "bar", data: ordered.value.map(m => m.R2),
             itemStyle: { color: "#4C72B0" } }],
}))

const url = api.staticUrl
</script>

<template>
  <h2>Client {{ lclid }}</h2>
  <p v-if="err" class="muted">{{ err }}</p>

  <div class="row">
    <div class="panel">
      <h3 style="margin-top:0">Metrics by method</h3>
      <VChart v-if="ordered.length" class="chart" :option="r2Option" autoresize />
      <table v-if="ordered.length" style="margin-top:12px">
        <thead><tr><th>Method</th><th>R2</th><th>MAE</th><th>RMSE</th><th>Spike_MAE</th></tr></thead>
        <tbody>
          <tr v-for="m in ordered" :key="m.method">
            <td>{{ m.method }}</td>
            <td>{{ m.R2?.toFixed(4) }}</td>
            <td>{{ m.MAE?.toFixed(4) }}</td>
            <td>{{ m.RMSE?.toFixed(4) }}</td>
            <td>{{ m.Spike_MAE?.toFixed(4) }}</td>
          </tr>
        </tbody>
      </table>
      <p v-else class="muted">No metrics found for this client.</p>
    </div>
  </div>

  <div class="row">
    <div class="panel">
      <h3 style="margin-top:0">EDA</h3>
      <img v-if="images.eda" class="artefact" :src="url(images.eda)" />
      <p v-else-if="pollLeft > 0" class="muted">Se generează EDA… (poate dura câteva secunde)</p>
      <p v-else class="muted">No EDA image.</p>
    </div>
  </div>
  <div class="row">
    <div class="panel">
      <h3 style="margin-top:0">Training curves</h3>
      <img v-if="images.training" class="artefact" :src="url(images.training)" />
      <p v-else class="muted">No training image.</p>
    </div>
    <div class="panel">
      <h3 style="margin-top:0">Predictions (actual vs predicted)</h3>
      <img v-if="images.predictions" class="artefact" :src="url(images.predictions)" />
      <p v-else class="muted">No prediction image.</p>
    </div>
  </div>
</template>
