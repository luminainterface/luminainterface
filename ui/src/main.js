import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import './styles/main.css'

/* --- Chart.js time-scale registration --- */
import {
  Chart,
  TimeScale,
  LinearScale,
  CategoryScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend
} from 'chart.js'
import 'chartjs-adapter-date-fns'

Chart.register(
  TimeScale,
  LinearScale,
  CategoryScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend
)
/* --------------------------------------- */

const app = createApp(App)
const pinia = createPinia()
app.use(pinia)
app.use(router)

import Toast, { POSITION } from 'vue-toastification'
import 'vue-toastification/dist/index.css'
import { notify } from '@/bus/notify'

app.use(Toast, { position: POSITION.BOTTOM_RIGHT })

notify.on('toast', ({ level, msg }) => {
  if (app.config.globalProperties.$toast && typeof app.config.globalProperties.$toast[level] === 'function') {
    app.config.globalProperties.$toast[level](msg)
  }
})

app.mount('#app') 