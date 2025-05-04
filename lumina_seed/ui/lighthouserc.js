module.exports = {
    ci: {
        collect: {
            startServerCommand: 'pnpm dev',
            url: ['http://localhost:3000'],
            numberOfRuns: 3,
        },
        assert: {
            assertions: {
                'categories:performance': ['error', { minScore: 0.90 }],
                'categories:accessibility': ['error', { minScore: 0.95 }],
                'categories:best-practices': ['error', { minScore: 0.95 }],
                'categories:seo': ['error', { minScore: 0.95 }],
                'first-contentful-paint': ['error', { maxNumericValue: 2000 }],
                'largest-contentful-paint': ['error', { maxNumericValue: 2500 }],
                'cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
                'total-blocking-time': ['error', { maxNumericValue: 300 }],
                'interactive': ['error', { maxNumericValue: 3500 }],
            },
        },
        upload: {
            target: 'temporary-public-storage',
        },
    },
}; 