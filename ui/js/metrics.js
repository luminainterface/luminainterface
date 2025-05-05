// System metrics monitoring
async function updateMetrics() {
    const metricsContainer = document.getElementById('metrics-container');
    
    try {
        // Fetch metrics from backend
        const response = await fetch('http://localhost:8000/metrics/summary');
        if (!response.ok) {
            throw new Error('Failed to fetch metrics');
        }
        
        const metrics = await response.json();
        
        // Clear existing metrics
        metricsContainer.innerHTML = '';
        
        // Create metric cards
        for (const [key, value] of Object.entries(metrics)) {
            const card = document.createElement('div');
            card.className = 'metric-card';
            
            const title = document.createElement('h3');
            title.className = 'text-sm font-medium text-gray-500 dark:text-gray-400';
            title.textContent = key.replace(/_/g, ' ').toUpperCase();
            
            const valueElement = document.createElement('div');
            valueElement.className = 'metric-value';
            valueElement.textContent = typeof value === 'number' 
                ? value.toFixed(2) 
                : value;
            
            card.appendChild(title);
            card.appendChild(valueElement);
            metricsContainer.appendChild(card);
        }
    } catch (error) {
        console.error('Error updating metrics:', error);
        
        const errorCard = document.createElement('div');
        errorCard.className = 'metric-card';
        errorCard.innerHTML = `
            <h3 class="text-sm font-medium text-red-500">Error</h3>
            <div class="metric-value text-red-500">Failed to load metrics</div>
        `;
        
        metricsContainer.appendChild(errorCard);
    }
}

// Initialize metrics update
document.addEventListener('DOMContentLoaded', () => {
    updateMetrics();
    // Update metrics every 5 seconds
    setInterval(updateMetrics, 5000);
}); 