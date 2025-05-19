# Build stage
FROM node:18-alpine AS build
WORKDIR /app
COPY . .
RUN npm ci
RUN npm run build

# Production stage
FROM nginx:1.25-alpine

# Create necessary directories
RUN mkdir -p /usr/share/nginx/html/css /usr/share/nginx/html/js

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy static files
COPY index.html /usr/share/nginx/html/
COPY 50x.html /usr/share/nginx/html/
COPY css/styles.css /usr/share/nginx/html/css/
COPY js/app.js /usr/share/nginx/html/js/
COPY js/metrics.js /usr/share/nginx/html/js/

# Set permissions
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html

# Copy built files from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80 (mapped to 3000 in docker-compose)
EXPOSE 80

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
  CMD wget --quiet --tries=1 --spider http://localhost:80/health || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"] 