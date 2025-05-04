// Import jest-axe
import { configureAxe } from 'jest-axe';

// Configure axe with custom rules
const axe = configureAxe({
    rules: {
        // Disable rules that don't apply to our use case
        'color-contrast': { enabled: true },
        'aria-allowed-attr': { enabled: true },
        'aria-required-attr': { enabled: true },
        'aria-valid-attr-value': { enabled: true },
        'aria-valid-attr': { enabled: true },
        'button-name': { enabled: true },
        'document-title': { enabled: true },
        'html-has-lang': { enabled: true },
        'image-alt': { enabled: true },
        'label': { enabled: true },
        'link-name': { enabled: true },
        'list': { enabled: true },
        'listitem': { enabled: true },
        'role-img-alt': { enabled: true }
    }
});

// Make axe available globally
global.axe = axe; 