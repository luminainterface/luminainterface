// Service health check
async function checkServiceHealth() {
    const services = ['hub-api', 'scheduler', 'redis', 'llm-engine', 'vector-db'];
    const healthContainer = document.getElementById('service-health');
    
    // Clear existing health indicators
    healthContainer.innerHTML = '';
    
    // Show error state for all services
    for (const service of services) {
        const indicator = document.createElement('div');
        indicator.className = 'health-indicator health-down';
        indicator.textContent = `${service}: unknown`;
        healthContainer.appendChild(indicator);
    }
}

/* ---------- helpers -------------------------------------------------- */
// flip to true when you want noisy stream logs
const DEBUG_STREAM = false;

function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

function parseChunk(txt) {
    try { return JSON.parse(txt); }
    catch { return null; }
}

async function safeJson(resp) {
    try { return await resp.json(); }
    catch { return null; }
}

function renderAssistantText(text) {
    const assistantMessage = document.createElement('div');
    assistantMessage.className = 'message assistant';
    assistantMessage.textContent = text;
    chatContainer.appendChild(assistantMessage);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Chat functionality
const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;
    
    // Add user message to chat
    const userMessage = document.createElement('div');
    userMessage.className = 'message user';
    userMessage.textContent = message;
    chatContainer.appendChild(userMessage);
    
    // Clear input
    messageInput.value = '';
    
    try {
        const apiKey = localStorage.getItem('lumina_api_key') || '';
        const response = await fetch('http://localhost:8000/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(apiKey && { 'X-API-Key': apiKey }),
            },
            body: JSON.stringify({
                messages: [{ role: "user", content: message }],
                stream: true,
                model: "phi"  // Use known working model
            }),
        });

        /* --- handle non-200 right away --- */
        if (!response.ok) {
            // 404 when model missing, 401 when key wrong, etc.
            const errJson = await safeJson(response);
            showToast(`${response.status}: ${errJson?.detail || response.statusText}`);
            throw new Error(`HTTP ${response.status}`);
        }

        /* --- decide: streaming vs plain JSON ------------------------------ */
        const ctype = response.headers.get('content-type') || '';
        if (ctype.startsWith('application/json')) {
            // backend gave us one blob → read, render, and exit
            const blob = await response.json();
            renderAssistantText(blob.choices?.[0]?.message?.content || '[blank]');
            return;
        }

        /* --- real SSE stream path ----------------------------------------- */
        const reader = response.body.getReader();
        const dec = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += dec.decode(value, { stream: true });

            // split by newline, process each chunk
            buffer = buffer.split('\n').reduce((acc, line) => {
                if (!line.trim()) return acc;             // skip empties
                if (!line.startsWith('data:')) return acc; // skip bad lines
                const json = parseChunk(line.slice(5));
                if (json) renderAssistantText(json.delta?.content || '');
                if (DEBUG_STREAM) console.log('[chunk]', json);
                return '';
            }, '');
        }
    } catch (error) {
        console.error('Error sending message:', error);
        if (!document.querySelector('.message.assistant:last-child')) {
            renderAssistantText('Sorry, there was an error processing your message.');
        }
    }
}

// Event listeners
sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

/* ---------- Health Overlay ------------------------------------------ */
const healthURL = "/health";
const overlay = document.getElementById("healthOverlay");
const COLOR_MAP = { ok: "bg-green-500", down: "bg-red-600", warn: "bg-yellow-500" };

async function pollHealth() {
  try {
    const res = await fetch(healthURL, { cache: "no-store" });
    const json = await res.json();          // { redis:"ok", qdrant:"ok", ollama:"down", scheduler:"ok" }
    renderHealthBadges(json);
  } catch (e) {
    renderHealthBadges({ hub: "down" });
  } finally {
    setTimeout(pollHealth, 30_000);
  }
}

function renderHealthBadges(obj) {
  overlay.innerHTML = "";                  // clear old
  const allOk = Object.values(obj).every(status => status === "ok");
  overlay.classList.toggle("hidden", allOk);  // hide if all services ok
  
  Object.entries(obj).forEach(([svc, status]) => {
    const span = document.createElement("span");
    span.className = `px-2 py-0.5 rounded-full ${COLOR_MAP[status] || "bg-gray-500"} animate-pulse`;
    span.textContent = `${svc}: ${status}`;
    overlay.appendChild(span);
  });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkServiceHealth();
    // Refresh health status every 30 seconds
    setInterval(checkServiceHealth, 30000);
    // Start health overlay polling
    pollHealth();
}); 