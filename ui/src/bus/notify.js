import mitt from 'mitt'
export const notify = mitt()

export const levels = {
  info:    'info',
  success: 'success',
  warn:    'warning',
  error:   'error',
  debug:   'default'
} 