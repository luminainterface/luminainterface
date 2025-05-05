# Lumina UI Setup Guide

This guide will help you set up and run the Lumina frontend application.

## Prerequisites

- Node.js 18.x or later
- npm 9.x or later
- Git

## Installation

1. Clone the repository (if you haven't already):
   ```bash
   git clone <repository-url>
   cd ui
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Development

### Running the Development Server

Start the development server with hot-reload:
```bash
npm run dev
```

The application will be available at `http://localhost:5173` by default.

### Type Checking

Run TypeScript type checking:
```bash
npm run type-check
```

### Building for Production

Build the application for production:
```bash
npm run build
```

The built files will be in the `dist` directory.

### Preview Production Build

Preview the production build locally:
```bash
npm run preview
```

## Testing

### Unit Tests

Run unit tests:
```bash
npm run test
```

Watch mode:
```bash
npm run test:watch
```

Generate coverage report:
```bash
npm run test:coverage
```

### E2E Tests with Cypress

Open Cypress Test Runner:
```bash
npm run cypress:open
```

Run Cypress tests headlessly:
```bash
npm run cypress:headless
```

Run specific test suites:
- Smoke tests: `npm run cypress:smoke`
- Wiki QA tests: `npm run cypress:stubbed`
- Live Wiki QA tests: `npm run cypress:live`

## Docker Deployment

Build and run the Docker container:
```bash
docker build -t lumina-ui .
docker run -p 80:80 lumina-ui
```

## Environment Configuration

The frontend connects to the following backend services:
- Graph API: `http://localhost:8201`
- MasterChat: `http://localhost:8301`
- Event Mux: `http://localhost:8101`
- Crawler: `http://localhost:8401`

Make sure these services are running before starting the frontend.

## Project Structure

```
ui/
├── src/              # Source files
├── cypress/          # E2E tests
├── public/           # Static assets
├── css/             # CSS files
├── js/              # JavaScript utilities
└── tests/           # Unit tests
```

## Key Dependencies

- Vue 3.4.x - Frontend framework
- TypeScript 5.3.x - Type safety
- Vite 5.0.x - Build tool
- TailwindCSS 3.4.x - Styling
- Pinia 2.1.x - State management
- Vue Router 4.2.x - Routing
- Chart.js 4.4.x - Data visualization
- D3.js 7.8.x - Advanced visualizations
- Socket.IO Client 4.7.x - Real-time communication

## Development Guidelines

1. **TypeScript**: Always use TypeScript for new code
2. **Components**: Follow Vue 3 Composition API patterns
3. **State Management**: Use Pinia for global state
4. **Styling**: Use TailwindCSS utility classes
5. **Testing**: Write tests for new features

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   - If port 5173 is in use, Vite will automatically try the next available port
   - Check the console output for the actual port being used

2. **TypeScript Errors**
   - Run `npm run type-check` to identify type issues
   - Make sure all dependencies are properly installed

3. **Build Failures**
   - Clear the `node_modules` directory and run `npm install` again
   - Check for version conflicts in `package.json`

### Getting Help

- Check the [Vue 3 Documentation](https://vuejs.org/)
- Review the [Vite Documentation](https://vitejs.dev/)
- Consult the [TailwindCSS Documentation](https://tailwindcss.com/)

## Contributing

1. Create a new branch for your feature
2. Write tests for new functionality
3. Follow the existing code style
4. Submit a pull request

## License

[Add your license information here] 