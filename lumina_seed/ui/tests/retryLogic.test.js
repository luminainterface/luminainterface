describe('Retry Logic', () => {
    let mockFetch;
    let mockResponse;
    let mockReader;
    let mockConsoleError;

    beforeEach(() => {
        // Mock fetch
        mockFetch = jest.fn();
        global.fetch = mockFetch;
        
        // Mock console.error
        mockConsoleError = jest.spyOn(console, 'error').mockImplementation(() => {});
        
        // Clear document body
        document.body.innerHTML = '';
    });

    afterEach(() => {
        // Restore console.error
        mockConsoleError.mockRestore();
    });

    test('successfully sends message on first try', async () => {
        // Mock successful response
        mockReader = {
            read: jest.fn().mockResolvedValueOnce({
                done: false,
                value: new TextEncoder().encode('Hello')
            }).mockResolvedValueOnce({
                done: true
            })
        };
        mockResponse = {
            ok: true,
            body: {
                getReader: () => mockReader
            }
        };
        mockFetch.mockResolvedValueOnce(mockResponse);

        const result = await retryMessage('Test message');
        expect(result).toBe(true);
        expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    test('retries on failure with exponential backoff', async () => {
        // Mock failed responses
        mockFetch
            .mockRejectedValueOnce(new Error('Network error'))
            .mockRejectedValueOnce(new Error('Network error'))
            .mockResolvedValueOnce({
                ok: true,
                body: {
                    getReader: () => ({
                        read: jest.fn().mockResolvedValueOnce({
                            done: true
                        })
                    })
                }
            });

        // Spy on setTimeout
        jest.useFakeTimers();

        const result = await retryMessage('Test message');
        
        // Check retry attempts
        expect(mockFetch).toHaveBeenCalledTimes(3);
        expect(result).toBe(true);

        // Check retry badge
        const retryBadge = document.querySelector('.retry-badge');
        expect(retryBadge).not.toBeInTheDocument();

        jest.useRealTimers();
    });

    test('shows error after max retries', async () => {
        // Mock all attempts failing
        mockFetch
            .mockRejectedValueOnce(new Error('Network error'))
            .mockRejectedValueOnce(new Error('Network error'))
            .mockRejectedValueOnce(new Error('Network error'));

        // Spy on setTimeout
        jest.useFakeTimers();

        const result = await retryMessage('Test message');
        
        // Check retry attempts
        expect(mockFetch).toHaveBeenCalledTimes(3);
        expect(result).toBe(false);

        // Check error toast
        const toast = document.querySelector('.fixed.top-4.right-4');
        expect(toast).toBeInTheDocument();
        expect(toast).toHaveTextContent('Failed to send message after multiple retries');

        jest.useRealTimers();
    });

    test('shows retry badge with countdown', async () => {
        // Mock first attempt failing
        mockFetch.mockRejectedValueOnce(new Error('Network error'));

        // Spy on setTimeout
        jest.useFakeTimers();

        retryMessage('Test message');
        
        // Check retry badge
        const retryBadge = document.querySelector('.retry-badge');
        expect(retryBadge).toBeInTheDocument();
        expect(retryBadge).toHaveTextContent('Retrying in 1s...');

        jest.useRealTimers();
    });
}); 