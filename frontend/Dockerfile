# CiRA ME - Frontend Dockerfile
# Multi-stage build for Vue.js 3 + Vite application

# Stage 1: Build stage
FROM node:20-alpine as builder

# Set working directory
WORKDIR /app

# Copy package files for dependency installation
# This layer is cached unless package.json or package-lock.json changes
COPY package*.json ./

# Install all dependencies (including devDependencies for build)
RUN npm install

# Copy application source code
COPY . .

# Build argument for API URL (empty because frontend code already includes /api prefix)
ARG VITE_API_URL=
ENV VITE_API_URL=$VITE_API_URL

# Build the application for production (skip type checking in Docker build)
RUN npx vite build

# Stage 2: Production stage with Nginx
FROM nginx:1.25-alpine as production

# Install curl for health checks
RUN apk add --no-cache curl

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built application from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Create non-root user for Nginx
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    chown -R nginx:nginx /etc/nginx/conf.d

# Create pid directory for nginx
RUN touch /var/run/nginx.pid && \
    chown -R nginx:nginx /var/run/nginx.pid

# Switch to non-root user
USER nginx

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:80/ || exit 1

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
