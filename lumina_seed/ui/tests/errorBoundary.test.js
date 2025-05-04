describe('Error Boundary', () => {
    let originalOnError;
    let mockConsoleError;

    beforeEach(() => {
        // Save original error handler
        originalOnError = window.onerror;
        // Mock console.error
        mockConsoleError = jest.spyOn(console, 'error').mockImplementation(() => {});
        // Reset error state
        hasError = false;
        // Clear document body
        document.body.innerHTML = '';
    });

    afterEach(() => {
        // Restore original error handler
        window.onerror = originalOnError;
        // Restore console.error
        mockConsoleError.mockRestore();
    });

    test('shows error boundary on unhandled error', () => {
        // Trigger error
        window.onerror('Test error', 'test.js', 1, 1, new Error('Test error'));
        
        // Check if error boundary is shown
        const errorDiv = document.querySelector('.fixed.inset-0');
        expect(errorDiv).toBeInTheDocument();
        
        // Check error message
        const errorMessage = errorDiv.querySelector('h2');
        expect(errorMessage).toHaveTextContent('Oops! Something went wrong');
        
        // Check refresh button
        const refreshButton = errorDiv.querySelector('button');
        expect(refreshButton).toBeInTheDocument();
        expect(refreshButton).toHaveTextContent('Refresh Page');
    });

    test('only shows error boundary once', () => {
        // Trigger first error
        window.onerror('First error', 'test.js', 1, 1, new Error('First error'));
        const firstErrorDiv = document.querySelector('.fixed.inset-0');
        
        // Trigger second error
        window.onerror('Second error', 'test.js', 1, 1, new Error('Second error'));
        const errorDivs = document.querySelectorAll('.fixed.inset-0');
        
        // Should only have one error boundary
        expect(errorDivs.length).toBe(1);
        expect(errorDivs[0]).toBe(firstErrorDiv);
    });

    test('preserves chat history on error', () => {
        // Mock localStorage
        const mockHistory = [
            { role: 'user', content: 'Test message', timestamp: Date.now() }
        ];
        localStorage.setItem = jest.fn();
        localStorage.getItem = jest.fn().mockReturnValue(JSON.stringify(mockHistory));
        
        // Trigger error
        window.onerror('Test error', 'test.js', 1, 1, new Error('Test error'));
        
        // Check if history is preserved
        expect(localStorage.getItem).toHaveBeenCalledWith('lumina_history');
    });
}); 