import mitt from 'mitt';

const bus = mitt();

export const initGraphSocket = () => {
  const url = import.meta.env.VITE_WS_URL || 'ws://localhost:8101/ws';
  const ws = new WebSocket(url);

  ws.onopen  = () => bus.emit('ws', 'OPEN');
  ws.onclose = () => { bus.emit('ws', 'CLOSED'); setTimeout(initGraphSocket, 2000); };

  ws.onmessage = e => {
    const data = JSON.parse(e.data);
    if (data.type === 'planner_log') bus.emit('planner', data.payload);
    if (data.type === 'edge.add')    bus.emit('graph',   data);
    if (data.type === 'assistant')   bus.emit('assistant', data.payload);
  };
};

export default bus; 