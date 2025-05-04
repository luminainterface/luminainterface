const { keymap, initKeyboardShortcuts } = require('../keymap');

describe('Keyboard Shortcuts', () => {
    let mockSendButton;
    let mockMessageInput;
    let mockLocalStorage;

    beforeEach(() => {
        // Mock DOM elements
        mockSendButton = {
            click: jest.fn(),
            disabled: false
        };
        mockMessageInput = {
            value: '',
            focus: jest.fn(),
            setSelectionRange: jest.fn()
        };
        
        document.getElementById = jest.fn((id) => {
            if (id === 'send-button') return mockSendButton;
            if (id === 'message-input') return mockMessageInput;
            return null;
        });
        
        // Mock localStorage
        mockLocalStorage = {
            getItem: jest.fn().mockReturnValue(JSON.stringify([
                { role: 'user', content: 'Last message', timestamp: Date.now() }
            ]))
        };
        Object.defineProperty(window, 'localStorage', {
            value: mockLocalStorage
        });
    });

    test('sends message on Cmd/Ctrl + Enter', () => {
        const event = new KeyboardEvent('keydown', {
            key: 'Enter',
            metaKey: true
        });
        
        keymap['mod+enter'](event);
        
        expect(event.preventDefault).toHaveBeenCalled();
        expect(mockSendButton.click).toHaveBeenCalled();
    });

    test('does not send message when button is disabled', () => {
        mockSendButton.disabled = true;
        
        const event = new KeyboardEvent('keydown', {
            key: 'Enter',
            metaKey: true
        });
        
        keymap['mod+enter'](event);
        
        expect(mockSendButton.click).not.toHaveBeenCalled();
    });

    test('edits last message on Up arrow when input is empty', () => {
        const event = new KeyboardEvent('keydown', {
            key: 'ArrowUp'
        });
        
        keymap['arrowup'](event);
        
        expect(event.preventDefault).toHaveBeenCalled();
        expect(mockMessageInput.value).toBe('Last message');
        expect(mockMessageInput.focus).toHaveBeenCalled();
        expect(mockMessageInput.setSelectionRange).toHaveBeenCalledWith(0, 'Last message'.length);
    });

    test('does not edit last message when input is not empty', () => {
        mockMessageInput.value = 'Current message';
        
        const event = new KeyboardEvent('keydown', {
            key: 'ArrowUp'
        });
        
        keymap['arrowup'](event);
        
        expect(event.preventDefault).not.toHaveBeenCalled();
        expect(mockMessageInput.value).toBe('Current message');
    });

    test('focuses textarea on Escape when not focused', () => {
        const event = new KeyboardEvent('keydown', {
            key: 'Escape'
        });
        
        keymap['escape'](event);
        
        expect(event.preventDefault).toHaveBeenCalled();
        expect(mockMessageInput.focus).toHaveBeenCalled();
    });

    test('does not focus textarea on Escape when already focused', () => {
        document.activeElement = mockMessageInput;
        
        const event = new KeyboardEvent('keydown', {
            key: 'Escape'
        });
        
        keymap['escape'](event);
        
        expect(event.preventDefault).not.toHaveBeenCalled();
        expect(mockMessageInput.focus).not.toHaveBeenCalled();
    });

    test('focuses search input on Cmd/Ctrl + K', () => {
        // Mock search input
        const mockSearchInput = {
            focus: jest.fn()
        };
        document.getElementById = jest.fn((id) => {
            if (id === 'search-input') return mockSearchInput;
            return null;
        });
        
        const event = new KeyboardEvent('keydown', {
            key: 'k',
            metaKey: true
        });
        
        keymap['mod+k'](event);
        
        expect(event.preventDefault).toHaveBeenCalled();
        expect(mockSearchInput.focus).toHaveBeenCalled();
    });

    test('does not focus search input when it does not exist', () => {
        // Mock getElementById to return null
        document.getElementById = jest.fn().mockReturnValue(null);
        
        const event = new KeyboardEvent('keydown', {
            key: 'k',
            metaKey: true
        });
        
        keymap['mod+k'](event);
        
        expect(event.preventDefault).toHaveBeenCalled();
        expect(document.getElementById).toHaveBeenCalledWith('search-input');
    });

    test('should focus search input on / key when not in input', () => {
        const searchInput = document.getElementById('search-input');
        const messageInput = document.getElementById('message-input');
        
        // Simulate / key press
        const event = new KeyboardEvent('keydown', { key: '/' });
        document.dispatchEvent(event);
        
        expect(document.activeElement).toBe(searchInput);
    });
    
    test('should not focus search input on / key when in input', () => {
        const searchInput = document.getElementById('search-input');
        const messageInput = document.getElementById('message-input');
        
        // Focus message input
        messageInput.focus();
        
        // Simulate / key press
        const event = new KeyboardEvent('keydown', { key: '/' });
        document.dispatchEvent(event);
        
        expect(document.activeElement).toBe(messageInput);
    });

    test('clears search input and focuses message input on Escape when search has content', () => {
        const mockSearchInput = {
            value: 'search query',
            focus: jest.fn()
        };
        const mockMessageInput = {
            focus: jest.fn()
        };
        
        document.getElementById = jest.fn((id) => {
            if (id === 'search-input') return mockSearchInput;
            if (id === 'message-input') return mockMessageInput;
            return null;
        });
        
        document.activeElement = mockSearchInput;
        
        const event = new KeyboardEvent('keydown', {
            key: 'Escape'
        });
        
        keymap['escape'](event);
        
        expect(event.preventDefault).toHaveBeenCalled();
        expect(mockSearchInput.value).toBe('');
        expect(mockMessageInput.focus).toHaveBeenCalled();
    });

    test('does not clear search input on Escape when search is empty', () => {
        const mockSearchInput = {
            value: '',
            focus: jest.fn()
        };
        const mockMessageInput = {
            focus: jest.fn()
        };
        
        document.getElementById = jest.fn((id) => {
            if (id === 'search-input') return mockSearchInput;
            if (id === 'message-input') return mockMessageInput;
            return null;
        });
        
        document.activeElement = mockSearchInput;
        
        const event = new KeyboardEvent('keydown', {
            key: 'Escape'
        });
        
        keymap['escape'](event);
        
        expect(event.preventDefault).toHaveBeenCalled();
        expect(mockSearchInput.value).toBe('');
        expect(mockMessageInput.focus).toHaveBeenCalled();
    });

    test('shows keyboard shortcuts modal on Shift + /', () => {
        // Mock modal element
        const mockModal = {
            classList: {
                remove: jest.fn()
            },
            querySelector: jest.fn()
        };
        const mockCloseButton = {
            focus: jest.fn()
        };
        
        document.getElementById = jest.fn((id) => {
            if (id === 'shortcuts-modal') return mockModal;
            return null;
        });
        mockModal.querySelector.mockReturnValue(mockCloseButton);
        
        const event = new KeyboardEvent('keydown', {
            key: '/',
            shiftKey: true
        });
        
        keymap['shift+/'](event);
        
        expect(event.preventDefault).toHaveBeenCalled();
        expect(mockModal.classList.remove).toHaveBeenCalledWith('hidden');
        expect(mockModal.querySelector).toHaveBeenCalledWith('button');
        expect(mockCloseButton.focus).toHaveBeenCalled();
    });

    test('handles missing modal gracefully', () => {
        document.getElementById = jest.fn().mockReturnValue(null);
        
        const event = new KeyboardEvent('keydown', {
            key: '/',
            shiftKey: true
        });
        
        // Should not throw
        expect(() => keymap['shift+/'](event)).not.toThrow();
    });
}); 