<script setup>
import { ref, watch, onMounted } from "vue"
import { RouterLink, RouterView, useRoute } from "vue-router"

const route    = useRoute()
const menuOpen = ref(false)
const theme    = ref("dark")

function applyTheme(t) {
  theme.value = t
  document.documentElement.setAttribute("data-theme", t)
  try { localStorage.setItem("theme", t) } catch {}
}
function toggleTheme() {
  applyTheme(theme.value === "dark" ? "light" : "dark")
}

onMounted(() => {
  theme.value = document.documentElement.getAttribute("data-theme") || "dark"
})

watch(() => route.fullPath, () => { menuOpen.value = false })
</script>

<template>
  <div class="app-shell">
    <header class="navbar">
      <button class="burger" @click="menuOpen = !menuOpen" aria-label="Toggle menu">☰</button>
      <span class="brand">⚡ FL Dashboard</span>

      <nav :class="{ open: menuOpen }">
        <RouterLink to="/">Overview</RouterLink>
        <RouterLink to="/live">Live Round</RouterLink>
        <RouterLink to="/clients">Clients</RouterLink>
        <RouterLink to="/comparison">Comparison</RouterLink>
      </nav>

      <span class="spacer"></span>

      <button class="theme-toggle ghost" @click="toggleTheme"
              :aria-label="theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'">
        {{ theme === "dark" ? "☀ Light" : "🌙 Dark" }}
      </button>
    </header>

    <main class="content">
      <RouterView />
    </main>
  </div>
</template>
