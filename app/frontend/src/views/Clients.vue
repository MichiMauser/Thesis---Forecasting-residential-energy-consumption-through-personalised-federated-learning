<script setup>
import { ref, onMounted } from "vue"
import { api } from "../api.js"

const clients = ref([])
const err = ref("")
onMounted(async () => {
  try { clients.value = await api.clients() } catch (e) { err.value = String(e) }
})
</script>

<template>
  <h2>Clients</h2>
  <p v-if="err" class="muted">{{ err }}</p>
  <div class="panel">
    <table>
      <thead>
        <tr><th>Client</th><th>Process</th><th>Last round</th><th>R2</th><th>MAE</th><th>Spike_MAE</th><th></th></tr>
      </thead>
      <tbody>
        <tr v-for="c in clients" :key="c.lclid">
          <td>{{ c.lclid }}</td>
          <td>
            <span class="dot" :class="c.process_up === true ? 'on' : (c.process_up === false ? 'off' : 'idle')"></span>
            {{ c.process_up === true ? "up" : (c.process_up === false ? "down" : "—") }}
          </td>
          <td>{{ c.last_seen_round ?? "—" }}</td>
          <td>{{ c.R2 != null ? c.R2.toFixed(4) : "—" }}</td>
          <td>{{ c.MAE != null ? c.MAE.toFixed(4) : "—" }}</td>
          <td>{{ c.Spike_MAE != null ? c.Spike_MAE.toFixed(4) : "—" }}</td>
          <td><RouterLink :to="`/clients/${c.lclid}`">detail →</RouterLink></td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
