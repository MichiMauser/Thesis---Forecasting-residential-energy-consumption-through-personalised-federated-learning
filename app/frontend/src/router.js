import { createRouter, createWebHistory } from "vue-router"

import Overview from "./views/Overview.vue"
import LiveRound from "./views/LiveRound.vue"
import Clients from "./views/Clients.vue"
import ClientDetail from "./views/ClientDetail.vue"
import Comparison from "./views/Comparison.vue"

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "overview", component: Overview, meta: { title: "Overview / Control" } },
    { path: "/live", name: "live", component: LiveRound, meta: { title: "Live Round" } },
    { path: "/clients", name: "clients", component: Clients, meta: { title: "Clients" } },
    { path: "/clients/:lclid", name: "client", component: ClientDetail, meta: { title: "Client" } },
    { path: "/comparison", name: "comparison", component: Comparison, meta: { title: "Comparison" } },
  ],
})
