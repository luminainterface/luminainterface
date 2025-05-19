import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from '../views/Dashboard.vue'
import FractalView from '../components/FractalView.vue'
import SubgraphView from '../components/SubgraphView.vue'
import MetricsPanel from '../components/MetricsPanel.vue'
import PlannerConsole from '../components/PlannerConsole.vue'
import ChatPanel from '../components/ChatPanel.vue'
import DualChatDashboard from '../views/DualChatDashboard.vue'

const routes = [
  {
    path: '/',
    name: 'dashboard',
    component: Dashboard
  },
  {
    path: '/graph',
    name: 'fractal',
    component: FractalView
  },
  {
    path: '/chat',
    name: 'chat',
    component: ChatPanel
  },
  {
    path: '/subgraph',
    name: 'subgraph',
    component: SubgraphView
  },
  {
    path: '/metrics',
    name: 'metrics',
    component: MetricsPanel
  },
  {
    path: '/planner',
    name: 'planner',
    component: PlannerConsole
  },
  {
    path: '/all-in-one',
    name: 'dualchatdashboard',
    component: DualChatDashboard
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router 