// Keyboard shortcut handler
const keymap = {
    // Send message with Cmd/Ctrl + Enter
    'mod+enter': (e) => {
        e.preventDefault();
        const sendButton = document.getElementById('send-button');
        if (!sendButton.disabled) {
            sendButton.click();
        }
    },
    
    // Edit last message with Up arrow
    'arrowup': (e) => {
        const messageInput = document.getElementById('message-input');
        if (messageInput.value.trim() === '') {
            e.preventDefault();
            const history = JSON.parse(localStorage.getItem('lumina_history') || '[]');
            const lastUserMessage = [...history].reverse().find(msg => msg.role === 'user');
            if (lastUserMessage) {
                messageInput.value = lastUserMessage.content;
                messageInput.focus();
                messageInput.setSelectionRange(0, messageInput.value.length);
            }
        }
    },
    
    // Focus textarea with Escape
    'escape': (e) => {
        const messageInput = document.getElementById('message-input');
        const searchInput = document.getElementById('search-input');
        
        // If search input is focused and has content, clear it
        if (document.activeElement === searchInput && searchInput.value.trim()) {
            e.preventDefault();
            searchInput.value = '';
            messageInput.focus();
            return;
        }
        
        // Otherwise, focus message input if not already focused
        if (document.activeElement !== messageInput) {
            e.preventDefault();
            messageInput.focus();
        }
    },

    // Focus search with Cmd/Ctrl + K
    'mod+k': (e) => {
        e.preventDefault();
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.focus();
        }
    },

    'Enter': {
        metaKey: true,
        action: () => {
            const sendButton = document.getElementById('send-button');
            if (sendButton && !sendButton.disabled) {
                sendButton.click();
            }
        }
    },
    'ArrowUp': {
        action: () => {
            const input = document.getElementById('message-input');
            if (input && !input.value.trim()) {
                const lastMessage = localStorage.getItem('lastMessage');
                if (lastMessage) {
                    input.value = lastMessage;
                    input.focus();
                }
            }
        }
    },
    'Escape': {
        action: () => {
            const input = document.getElementById('message-input');
            if (input && document.activeElement !== input) {
                input.focus();
            }
        }
    },
    '/': {
        action: (e) => {
            // Only trigger if not typing in an input
            if (!['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
                e.preventDefault();
                const searchInput = document.getElementById('search-input');
                if (searchInput) {
                    searchInput.focus();
                }
            }
        }
    },
    // Show keyboard shortcuts modal with Shift + /
    'shift+/': (e) => {
        e.preventDefault();
        const modal = document.getElementById('shortcuts-modal');
        if (modal) {
            modal.classList.remove('hidden');
            // Focus the close button for keyboard navigation
            const closeButton = modal.querySelector('button');
            if (closeButton) closeButton.focus();
        }
    }
};

// Initialize keyboard shortcuts
function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Check for modifier keys
        const isMod = e.metaKey || e.ctrlKey;
        const key = e.key.toLowerCase();
        
        // Handle Cmd/Ctrl + Enter
        if (isMod && key === 'enter') {
            keymap['mod+enter'](e);
            return;
        }
        
        // Handle Cmd/Ctrl + K
        if (isMod && key === 'k') {
            keymap['mod+k'](e);
            return;
        }
        
        // Handle other shortcuts
        if (keymap[key]) {
            keymap[key](e);
        }
    });
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { keymap, initKeyboardShortcuts };
} 