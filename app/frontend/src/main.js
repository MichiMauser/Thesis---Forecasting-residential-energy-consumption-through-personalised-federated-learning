import { createApp } from "vue"
import VChart from "vue-echarts"

import "./echarts.js"
import "./style.css"
import App from "./App.vue"
import router from "./router.js"

createApp(App)
  .component("VChart", VChart)
  .use(router)
  .mount("#app")
