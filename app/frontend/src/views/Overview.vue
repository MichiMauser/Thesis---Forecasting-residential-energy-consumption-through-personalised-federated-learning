<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue"
import { useRouter } from "vue-router"
import { api } from "../api.js"

const router = useRouter()
const status = ref(null)
const clients = ref([])
const numRounds = ref(5)
const localEpochs = ref(5)
const finetuneEpochs = ref(10)
const method = ref("perfedavg")
const mu = ref(0.1)
const device = ref("cpu")
const busy = ref("")
const msg = ref("")
let timer = null

const available = ref([])
const selected = ref([])
const search = ref("")
const fedMsg = ref("")
const fedBusy = ref(false)
const maxClients = ref(20)
const editable = computed(() => !status.value?.daemons_up)   // only while down

const selectedSet = computed(() => new Set(selected.value))
const filtered = computed(() => {
  const q = search.value.trim().toUpperCase()
  if (!q) return []
  return available.value
    .filter(c => !selectedSet.value.has(c.lclid) &&
                 (c.lclid.toUpperCase().includes(q) ||
                  (c.acorn || "").toUpperCase().includes(q)))
    .slice(0, 40)
})

async function refresh() {
  try {
    status.value = await api.flStatus()
    clients.value = await api.clients()
  } catch (e) { msg.value = String(e) }
}

async function loadFederation() {
  try {
    const [avail, fed] = await Promise.all([api.availableClients(), api.getFederation()])
    available.value = avail
    selected.value = fed.clients
    maxClients.value = fed.max ?? 20
  } catch (e) { fedMsg.value = String(e) }
}

function addClient(lclid) {
  if (!selectedSet.value.has(lclid) && selected.value.length < maxClients.value) {
    selected.value = [...selected.value, lclid]
  }
}
function removeClient(lclid) {
  selected.value = selected.value.filter(c => c !== lclid)
}

async function saveFederation() {
  fedBusy.value = true; fedMsg.value = ""
  try {
    const r = await api.setFederation(selected.value)
    selected.value = r.clients
    fedMsg.value = `Saved ${r.clients.length} clients.`
    refresh()
  } catch (e) { fedMsg.value = String(e).replace(/^Error:\s*/, "") }
  finally { fedBusy.value = false }
}

async function act(name, fn) {
  busy.value = name; msg.value = ""
  try {
    const r = await fn()
    if (r.error) msg.value = r.error
    if (name === "run" && r.run_id) {
      router.push({ path: "/live", query: { run_id: r.run_id } })
    }
  } catch (e) { msg.value = String(e) }
  finally { busy.value = ""; refresh() }
}

onMounted(() => { refresh(); loadFederation(); timer = setInterval(refresh, 3000) })
onUnmounted(() => clearInterval(timer))
</script>

<template>
  <h2>Overview / Control</h2>

  <div class="panel">
    <div class="cards">
      <div class="card">
        <div class="lbl">Daemons</div>
        <div class="val">
          <span class="dot" :class="status?.daemons_up ? 'on' : 'off'"></span>
          {{ status?.daemons_up ? "up" : "down" }}
        </div>
      </div>
      <div class="card">
        <div class="lbl">Run</div>
        <div class="val">
          <span class="dot" :class="status?.run_active ? 'on' : 'idle'"></span>
          {{ status?.run_active ? "running" : "idle" }}
        </div>
      </div>
      <div class="card">
        <div class="lbl">Clients (min-clients)</div>
        <div class="val">{{ selected.length || "—" }}</div>
      </div>
      <div class="card">
        <div class="lbl">Current run_id</div>
        <div class="val">{{ status?.run_id || status?.latest_run_id || "—" }}</div>
      </div>
    </div>
  </div>

  <div class="panel">
    <h3 style="margin-top:0">Control</h3>
    <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap">
      <button class="ghost" :disabled="busy || status?.daemons_up"
              @click="act('up', api.flUp)">
        {{ busy === 'up' ? 'Starting…' : (status?.daemons_up ? 'Daemons up' : 'Start daemons (up)') }}
      </button>
      <button class="danger" :disabled="busy || !status?.daemons_up"
              @click="act('down', api.flDown)">
        {{ busy === 'down' ? 'Stopping…' : 'Stop daemons (down)' }}
      </button>
      <span style="width:18px"></span>
      <label>method</label>
      <select v-model="method">
        <option value="fedavg">FedAvg</option>
        <option value="fedprox">FedProx</option>
        <option value="perfedavg">PerFedAvg</option>
      </select>
      <template v-if="method === 'fedprox'">
        <label>mu</label>
        <input type="number" v-model.number="mu" min="0" step="0.01" style="width:70px" />
      </template>
      <label>device</label>
      <select v-model="device" title="CPU avoids GPU contention when many clients run at once">
        <option value="cpu">CPU</option>
        <option value="cuda">GPU (CUDA)</option>
        <option value="auto">Auto</option>
      </select>
      <label>rounds</label>
      <input type="number" v-model.number="numRounds" min="1" />
      <label>local epochs</label>
      <input type="number" v-model.number="localEpochs" min="1" />
      <template v-if="method === 'perfedavg'">
        <label>finetune epochs</label>
        <input type="number" v-model.number="finetuneEpochs" min="1" />
      </template>
      <button :disabled="busy || !status?.daemons_up || status?.run_active"
              @click="act('run', () => api.flRun({ method, mu, device, num_rounds: numRounds, local_epochs: localEpochs, finetune_epochs: finetuneEpochs }))">
        {{ busy === 'run' ? 'Launching…' : 'Trigger run' }}
      </button>
    </div>
    <p v-if="msg" class="muted" style="margin-bottom:0">{{ msg }}</p>
    <p v-else-if="!status?.daemons_up" class="muted" style="margin-bottom:0">
      Start the daemons before triggering a run. <code>min-clients</code> is set automatically to the {{ selected.length }} selected household(s).
    </p>
  </div>

  <div class="panel">
    <h3 style="margin-top:0">Federation setup
      <span class="muted" style="font-size:13px">— choose which households take part ({{ selected.length }}/{{ maxClients }})</span>
    </h3>
    <p v-if="!editable" class="muted" style="margin-top:0">
      Stop the daemons to change the client set (one SuperNode is started per client).
    </p>

    <div style="margin-bottom:8px">
      <span v-for="c in selected" :key="c" class="pill" style="text-transform:none">
        {{ c }}
        <a v-if="editable" href="#" style="margin-left:4px" @click.prevent="removeClient(c)">✕</a>
      </span>
      <span v-if="!selected.length" class="muted">No clients selected.</span>
    </div>

    <template v-if="editable">
      <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:8px">
        <input type="text" v-model="search" placeholder="search LCLid or Acorn…"
               style="background:var(--panel-2);color:var(--text);border:1px solid var(--border);padding:8px;border-radius:6px;width:240px" />
        <button :disabled="fedBusy || selected.length < 2" @click="saveFederation">
          {{ fedBusy ? 'Saving…' : 'Save selection' }}
        </button>
        <span v-if="fedMsg" class="muted">{{ fedMsg }}</span>
      </div>
      <table v-if="filtered.length">
        <thead><tr><th>LCLid</th><th>Acorn</th><th>Block</th><th></th></tr></thead>
        <tbody>
          <tr v-for="c in filtered" :key="c.lclid">
            <td>{{ c.lclid }}</td>
            <td>{{ c.acorn }}</td>
            <td>{{ c.file }}</td>
            <td><button class="ghost" style="padding:2px 10px"
                        :disabled="selected.length >= maxClients"
                        @click="addClient(c.lclid)">+</button></td>
          </tr>
        </tbody>
      </table>
      <p v-else-if="search" class="muted">No matches.</p>
      <p v-else class="muted">Type to search {{ available.length }} households.</p>
    </template>
  </div>

  <div class="panel">
    <h3 style="margin-top:0">Connected clients</h3>
    <div class="cards">
      <div class="card" v-for="c in clients" :key="c.lclid">
        <div class="lbl">
          <span class="dot" :class="c.process_up === true ? 'on' : (c.process_up === false ? 'off' : 'idle')"></span>
          {{ c.lclid }}
        </div>
        <div class="val">R2 {{ c.R2 != null ? c.R2.toFixed(3) : "—" }}</div>
        <div class="muted" style="font-size:12px">
          round {{ c.last_seen_round ?? "—" }} · drift {{ c.drift != null ? c.drift.toFixed(2) : "—" }}
        </div>
      </div>
    </div>
  </div>
</template>
