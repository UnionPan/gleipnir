# Multi-stage build for React frontend
FROM node:24-alpine AS builder

WORKDIR /app

# Copy package files
COPY frontend/package*.json ./

# Install exactly the locked dependencies (the lockfile is resolved with legacy peer deps)
RUN npm ci --legacy-peer-deps

# Copy source code
COPY frontend/ ./

# Build the app
RUN npm run build

# Production stage with nginx
FROM nginx:alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config
COPY deployment/docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copy built app from builder
COPY --from=builder /app/build /usr/share/nginx/html

# Set proper permissions
RUN chmod -R 755 /usr/share/nginx/html && \
    chown -R nginx:nginx /usr/share/nginx/html

# Expose port (Cloud Run uses 8080)
EXPOSE 8080

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
