# Lumina Chat UI

[![Post-Alpha Backlog](https://img.shields.io/badge/Post--Alpha%20Backlog-View%20Roadmap-blue)](POST_ALPHA_BACKLOG.md)

A modern, responsive chat interface for the Lumina AI assistant. Built with vanilla HTML/JS and Tailwind CSS.

## Features

- 💬 Real-time chat with streaming responses
- 📊 Live system metrics and service health monitoring
- 🎨 Dark mode support
- 📱 Mobile-first responsive design
- ⌨️ Keyboard shortcuts for power users
- 🔄 Automatic retry with visual feedback
- 💾 Message history persistence
- 🎯 Error boundary for graceful failure handling
- 🔍 Search conversations (coming soon)

## API Configuration

The UI requires an API key for authentication. Configure it in one of two ways:

1. **Environment Variable** (recommended for development):
   ```bash
   LUMINA_API_KEY=your_api_key_here
   ```

2. **Local Storage** (for testing):
   ```javascript
   localStorage.setItem('lumina_api_key', 'your_api_key_here');
   ```

The API key is automatically included in the `X-API-Key` header for all requests.

> **Note**: See the backend README for API key generation and management.

## Monitoring & Observability

The UI integrates with backend monitoring:

1. **Service Health**
   - Real-time status of Redis, Qdrant, and LLM services
   - Latency bars showing p95 response times
   - Automatic health checks every 30 seconds

2. **Rate Limiting**
   - Visual feedback for 429 responses
   - Automatic retry with exponential backoff
   - Rate limit status in service badges
   - Default: 10 requests per minute per key

3. **Metrics Dashboard**
   - Request latency histograms
   - Rate limit blocks vs. hits
   - Service health status
   - Vector pruning metrics

## Error Recovery

The UI includes several layers of error handling:

1. **Retry Badge**: When a message fails to send, a retry badge appears with a countdown timer. The system will automatically retry up to 3 times with exponential backoff.

2. **Error Boundary**: If an unhandled error occurs, an error boundary overlay appears with a refresh button. Chat history is preserved in localStorage.

3. **Toast Notifications**: Temporary notifications for errors and success states.

## Development

1. Install dependencies:
```bash
pnpm install
```

2. Start the development server:
```bash
pnpm dev
```

3. Run tests:
```bash
pnpm test
```

## Building for Production

```bash
pnpm build
```

The built files will be in the `dist` directory.

## Testing

The UI includes comprehensive test coverage for:
- Error boundary handling
- Retry logic
- Keyboard shortcuts
- Message history persistence
- Service health monitoring
- Rate limit handling

Run the test suite with:
```bash
pnpm test
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 