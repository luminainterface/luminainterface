import axios from 'axios'
import { notify } from '@/bus/notify'

const api = axios.create({ baseURL: import.meta.env.VITE_API_URL })

api.interceptors.request.use(conf=>{
  notify.emit('toast',{level:'debug',msg:`➡ ${conf.method.toUpperCase()} ${conf.url}`})
  return conf
})
api.interceptors.response.use(res=>{
  notify.emit('toast',{level:'success',msg:`⬅ ${res.config.url} ${res.status}`})
  return res
}, err=>{
  notify.emit('toast',{level:'error',msg:`❌ ${err.config?.url} ${err.message}`})
  return Promise.reject(err)
})

export default api 