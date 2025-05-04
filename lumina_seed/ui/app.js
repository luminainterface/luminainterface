// Error boundary state
let hasError = false;
const MAX_RETRIES = 3;
const RETRY_DELAYS = [1000, 3000, 5000]; // Exponential backoff delays

// Error boundary component
function showErrorBoundary() {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50';
    errorDiv.innerHTML = `
        <div class="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-xl max-w-md w-full mx-4">
            <h2 class="text-xl font-bold text-red-600 dark:text-red-400 mb-4">Oops! Something went wrong</h2>
            <p class="text-gray-600 dark:text-gray-300 mb-4">
                The application encountered an unexpected error. Your chat history is safe and will be restored when you refresh.
            </p>
            <button onclick="window.location.reload()" 
                class="w-full bg-primary hover:bg-blue-600 text-white px-4 py-2 rounded-lg shadow-sm hover:shadow transition-shadow focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary">
                Refresh Page
            </button>
        </div>
    `;
    document.body.appendChild(errorDiv);
}

// Global error handler
window.onerror = function(message, source, lineno, colno, error) {
    console.error('Global error:', error);
    if (!hasError) {
        hasError = true;
        showErrorBoundary();
    }
    return true;
};

// Initialize marked with highlight.js
marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    }
});

// Toast notification system
function showToast(message, type = 'error') {
    const toast = document.createElement('div');
    toast.className = `fixed top-4 right-4 px-4 py-2 rounded-lg shadow-lg transform transition-transform duration-300 translate-y-0 ${
        type === 'error' ? 'bg-red-900 text-red-100' : 'bg-green-900 text-green-100'
    }`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.transform = 'translateY(-100%)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// DOM Elements
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const chatMessages = document.getElementById('chat-messages');
const serviceHealth = document.getElementById('service-health');
const conversationCount = document.getElementById('conversation-count');
const tokenUsage = document.getElementById('token-usage');

// Loading state management
let isLoading = false;
function setLoading(loading) {
    isLoading = loading;
    sendButton.disabled = loading;
    sendButton.innerHTML = loading ? 
        '<div class="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full"></div>' : 
        'Send';
}

// Chart initialization
const ctx = document.getElementById('metrics-chart').getContext('2d');
const metricsChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Conversations',
                data: [],
                borderColor: '#3B82F6',
                tension: 0.4
            },
            {
                label: 'Token Usage',
                data: [],
                borderColor: '#10B981',
                tension: 0.4
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#9CA3AF'
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: '#374151'
                },
                ticks: {
                    color: '#9CA3AF'
                }
            },
            x: {
                grid: {
                    color: '#374151'
                },
                ticks: {
                    color: '#9CA3AF'
                }
            }
        }
    }
});

// Message history management
const HISTORY_KEY = 'lumina_history';
const MAX_HISTORY = 50;

function saveMessage(role, content) {
    const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    history.push({ role, content, timestamp: Date.now() });
    
    // Keep only last MAX_HISTORY messages
    if (history.length > MAX_HISTORY) {
        history.shift();
    }
    
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

function loadHistory() {
    const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    history.forEach(msg => appendMessage(msg.role, msg.content));
}

function clearHistory() {
    localStorage.removeItem(HISTORY_KEY);
    chatMessages.innerHTML = '';
    showToast('Chat history cleared', 'success');
}

// Add clear history button to header
const header = document.querySelector('header div');
const clearButton = document.createElement('button');
clearButton.className = 'text-sm text-gray-400 hover:text-gray-200';
clearButton.textContent = 'Clear History';
clearButton.onclick = clearHistory;
header.appendChild(clearButton);

// Update appendMessage to save to history
function appendMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `p-4 rounded-lg ${
        role === 'user' ? 'bg-blue-900 ml-12' : 
        role === 'assistant' ? 'bg-gray-800 mr-12' : 
        'bg-red-900'
    }`;
    
    if (role === 'assistant') {
        messageDiv.innerHTML = marked.parse(content);
    } else {
        messageDiv.textContent = content;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Save to history if not an error message
    if (role !== 'error') {
        saveMessage(role, content);
    }
}

// Service health monitoring
async function updateServiceHealth() {
    try {
        const response = await fetch('/modules');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const services = await response.json();
        
        // Group services by type
        const llmServices = services.filter(s => s.name.startsWith('llm-'));
        const storageServices = services.filter(s => ['redis', 'qdrant'].includes(s.name));
        const otherServices = services.filter(s => !llmServices.includes(s) && !storageServices.includes(s));
        
        // Create service groups HTML
        const serviceGroups = [
            { title: 'LLM Services', services: llmServices },
            { title: 'Storage', services: storageServices },
            { title: 'Other', services: otherServices }
        ].filter(group => group.services.length > 0);
        
        serviceHealth.innerHTML = serviceGroups.map(group => `
            <div class="flex flex-col gap-2">
                <h3 class="text-xs text-gray-400">${group.title}</h3>
                <div class="flex gap-2">
                    ${group.services.map(service => `
                        <div class="flex items-center gap-1 px-2 py-1 rounded-full text-sm
                            ${service.status === 'healthy' ? 'bg-green-900 text-green-300' : 
                              service.status === 'degraded' ? 'bg-yellow-900 text-yellow-300' : 
                              'bg-red-900 text-red-300'}"
                            role="status"
                            aria-label="${service.name} is ${service.status}">
                            <span class="w-2 h-2 rounded-full
                                ${service.status === 'healthy' ? 'bg-green-500' : 
                                  service.status === 'degraded' ? 'bg-yellow-500' : 
                                  'bg-red-500'}"></span>
                            ${service.name}
                            ${service.metrics ? `
                                <div class="flex items-center gap-1">
                                    <div class="w-12 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                                        <div class="h-full bg-current opacity-50" 
                                             style="width: ${Math.min(100, (service.metrics.latency / 1000) * 100)}%"></div>
                                    </div>
                                    <span class="text-xs opacity-75">${service.metrics.latency}ms</span>
                                </div>
                            ` : ''}
                        </div>
                    `).join('')}
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Failed to fetch service health:', error);
        showToast('Failed to fetch service health');
    }
}

// Update metrics
async function updateMetrics() {
    try {
        const response = await fetch('/metrics/summary');
        if (!response.ok) throw new Error('Failed to fetch metrics');
        const data = await response.json();
        
        // Hide skeletons
        document.getElementById('metrics-chart-skeleton').style.display = 'none';
        document.getElementById('conversation-count-skeleton').style.display = 'none';
        document.getElementById('token-usage-skeleton').style.display = 'none';
        
        // Show actual content
        document.getElementById('metrics-chart').style.display = 'block';
        document.getElementById('conversation-count').style.display = 'block';
        document.getElementById('token-usage').style.display = 'block';
        
        // Update metrics
        conversationCount.textContent = data.conversation_count;
        tokenUsage.textContent = data.token_usage;
        
        // Update chart
        metricsChart.data.labels.push(new Date().toLocaleTimeString());
        metricsChart.data.datasets[0].data.push(data.conversation_count);
        metricsChart.data.datasets[1].data.push(data.token_usage);
        
        // Keep only last 10 data points
        if (metricsChart.data.labels.length > 10) {
            metricsChart.data.labels.shift();
            metricsChart.data.datasets[0].data.shift();
            metricsChart.data.datasets[1].data.shift();
        }
        
        metricsChart.update();
    } catch (error) {
        console.error('Error updating metrics:', error);
        showToast('Failed to update metrics', 'error');
    }
}

// Retry logic for failed messages
async function retryMessage(message, retryCount = 0) {
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const reader = response.body.getReader();
        let assistantMessage = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = new TextDecoder().decode(value);
            assistantMessage += chunk;
            updateAssistantMessage(assistantMessage);
        }
        
        // Clear retry badge if successful
        const retryBadge = document.querySelector('.retry-badge');
        if (retryBadge) retryBadge.remove();
        
        return true;
    } catch (error) {
        console.error(`Retry attempt ${retryCount + 1} failed:`, error);
        
        if (retryCount < MAX_RETRIES - 1) {
            // Show retry badge
            const retryBadge = document.createElement('div');
            retryBadge.className = 'retry-badge fixed bottom-4 right-4 bg-yellow-500 text-white px-4 py-2 rounded-lg shadow-lg z-50';
            retryBadge.textContent = `Retrying in ${RETRY_DELAYS[retryCount]/1000}s...`;
            document.body.appendChild(retryBadge);
            
            // Wait and retry
            await new Promise(resolve => setTimeout(resolve, RETRY_DELAYS[retryCount]));
            return retryMessage(message, retryCount + 1);
        } else {
            // Show final error
            showToast('Failed to send message after multiple retries. Please try again later.', 'error');
            const retryBadge = document.querySelector('.retry-badge');
            if (retryBadge) retryBadge.remove();
            return false;
        }
    }
}

// Update sendMessage to use retry logic
async function sendMessage() {
    if (isLoading) return;
    const message = messageInput.value.trim();
    if (!message) return;
    
    setLoading(true);
    
    // Add user message to chat
    appendMessage('user', message);
    messageInput.value = '';
    
    try {
        await retryMessage(message);
    } catch (error) {
        console.error('Failed to send message:', error);
        showToast('Failed to send message. Please try again.');
    } finally {
        setLoading(false);
    }
}

function updateAssistantMessage(content) {
    const lastMessage = chatMessages.lastElementChild;
    if (lastMessage && lastMessage.classList.contains('bg-gray-800')) {
        lastMessage.innerHTML = marked.parse(content) + 
            '<div class="inline-block ml-2 animate-spin h-4 w-4 border-2 border-gray-400 border-t-transparent rounded-full"></div>';
    } else {
        appendMessage('assistant', content);
    }
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Event listeners
sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Initialize
loadHistory(); // Load history before other initializations
updateServiceHealth();
updateMetrics();

// Set up intervals
setInterval(updateServiceHealth, 30000); // Every 30 seconds
setInterval(updateMetrics, 10000); // Every 10 seconds

// Initially hide chart and show skeleton
document.getElementById('metrics-chart').style.display = 'none'; 