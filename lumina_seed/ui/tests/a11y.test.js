const { axe, toHaveNoViolations } = require('jest-axe');
expect.extend(toHaveNoViolations);

describe('Accessibility Tests', () => {
    let container;

    beforeEach(() => {
        // Load the HTML file
        document.body.innerHTML = require('fs').readFileSync(
            require('path').resolve(__dirname, '../index.html'),
            'utf8'
        );
        container = document.body;
    });

    afterEach(() => {
        document.body.innerHTML = '';
    });

    test('Main page should have no accessibility violations', async () => {
        const results = await axe(container);
        expect(results).toHaveNoViolations();
    });

    test('Chat interface should have no accessibility violations', async () => {
        const chatInterface = document.querySelector('main');
        const results = await axe(chatInterface);
        expect(results).toHaveNoViolations();
    });

    test('Service health badges should have no accessibility violations', async () => {
        const serviceHealth = document.getElementById('service-health');
        const results = await axe(serviceHealth);
        expect(results).toHaveNoViolations();
    });

    test('Metrics sidebar should have no accessibility violations', async () => {
        const metricsSidebar = document.querySelector('aside');
        const results = await axe(metricsSidebar);
        expect(results).toHaveNoViolations();
    });

    test('Message input should have no accessibility violations', async () => {
        const messageInput = document.getElementById('message-input');
        const results = await axe(messageInput);
        expect(results).toHaveNoViolations();
    });

    test('Send button should have no accessibility violations', async () => {
        const sendButton = document.getElementById('send-button');
        const results = await axe(sendButton);
        expect(results).toHaveNoViolations();
    });

    test('Search input should have no accessibility violations', async () => {
        const searchInput = document.getElementById('search-input');
        const results = await axe(searchInput);
        expect(results).toHaveNoViolations();
    });

    test('Error boundary should have no accessibility violations', async () => {
        // Simulate error boundary
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
        
        const results = await axe(errorDiv);
        expect(results).toHaveNoViolations();
    });

    test('Toast notifications should have no accessibility violations', async () => {
        // Simulate toast
        const toast = document.createElement('div');
        toast.className = 'fixed top-4 right-4 px-4 py-2 rounded-lg shadow-lg bg-red-900 text-red-100';
        toast.textContent = 'Error message';
        toast.setAttribute('role', 'alert');
        document.body.appendChild(toast);
        
        const results = await axe(toast);
        expect(results).toHaveNoViolations();
    });

    test('Service health groups should have no accessibility violations', async () => {
        // Simulate service health data
        const serviceHealth = document.getElementById('service-health');
        serviceHealth.innerHTML = `
            <div class="flex flex-col gap-2">
                <h3 class="text-xs text-gray-400">LLM Services</h3>
                <div class="flex gap-2">
                    <div class="flex items-center gap-1 px-2 py-1 rounded-full text-sm bg-green-900 text-green-300" role="status" aria-label="llm-service is healthy">
                        <span class="w-2 h-2 rounded-full bg-green-500"></span>
                        llm-service
                        <span class="text-xs opacity-75">(50ms)</span>
                    </div>
                </div>
            </div>
            <div class="flex flex-col gap-2">
                <h3 class="text-xs text-gray-400">Storage</h3>
                <div class="flex gap-2">
                    <div class="flex items-center gap-1 px-2 py-1 rounded-full text-sm bg-yellow-900 text-yellow-300" role="status" aria-label="redis is degraded">
                        <span class="w-2 h-2 rounded-full bg-yellow-500"></span>
                        redis
                        <span class="text-xs opacity-75">(100ms)</span>
                    </div>
                    <div class="flex items-center gap-1 px-2 py-1 rounded-full text-sm bg-green-900 text-green-300" role="status" aria-label="qdrant is healthy">
                        <span class="w-2 h-2 rounded-full bg-green-500"></span>
                        qdrant
                        <span class="text-xs opacity-75">(75ms)</span>
                    </div>
                </div>
            </div>
        `;
        
        const results = await axe(serviceHealth);
        expect(results).toHaveNoViolations();
    });
}); 