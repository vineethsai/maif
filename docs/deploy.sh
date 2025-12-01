#!/bin/bash

# MAIF Documentation Deployment Script
# This script automates the deployment of MAIF documentation to various platforms

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$DOCS_DIR/.vitepress/dist"
NODE_VERSION="18"

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Node.js version
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed. Please install Node.js $NODE_VERSION or higher."
        exit 1
    fi
    
    NODE_CURRENT=$(node --version | sed 's/v//')
    NODE_MAJOR=$(echo $NODE_CURRENT | cut -d. -f1)
    
    if [ "$NODE_MAJOR" -lt "$NODE_VERSION" ]; then
        log_error "Node.js version $NODE_CURRENT is too old. Please upgrade to version $NODE_VERSION or higher."
        exit 1
    fi
    
    log_success "Node.js version $NODE_CURRENT is compatible"
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        log_error "npm is not installed. Please install npm."
        exit 1
    fi
    
    log_success "npm is available"
    
    # Check git
    if ! command -v git &> /dev/null; then
        log_error "git is not installed. Please install git."
        exit 1
    fi
    
    log_success "git is available"
}

install_dependencies() {
    log_info "Installing dependencies..."
    
    cd "$DOCS_DIR"
    
    # Clean install
    if [ -d "node_modules" ]; then
        log_info "Removing existing node_modules..."
        rm -rf node_modules
    fi
    
    if [ -f "package-lock.json" ]; then
        log_info "Using package-lock.json for reproducible builds..."
        npm ci
    else
        log_warning "No package-lock.json found, using npm install..."
        npm install
    fi
    
    log_success "Dependencies installed successfully"
}

run_quality_checks() {
    log_info "Running quality checks..."
    
    cd "$DOCS_DIR"
    
    # Linting
    log_info "Running ESLint..."
    npm run lint
    log_success "Linting passed"
    
    # Type checking
    log_info "Running TypeScript type checking..."
    npm run type-check
    log_success "Type checking passed"
    
    # Link checking (if available)
    if command -v markdown-link-check &> /dev/null; then
        log_info "Checking markdown links..."
        find . -name "*.md" -not -path "./node_modules/*" -exec markdown-link-check {} \; || true
        log_success "Link checking completed"
    else
        log_warning "markdown-link-check not installed, skipping link validation"
    fi
}

build_documentation() {
    log_info "Building documentation..."
    
    cd "$DOCS_DIR"
    
    # Clean previous build
    if [ -d "$BUILD_DIR" ]; then
        log_info "Removing previous build..."
        rm -rf "$BUILD_DIR"
    fi
    
    # Build
    npm run build
    
    if [ ! -d "$BUILD_DIR" ]; then
        log_error "Build failed - output directory not found"
        exit 1
    fi
    
    # Verify build output
    if [ ! -f "$BUILD_DIR/index.html" ]; then
        log_error "Build failed - index.html not found"
        exit 1
    fi
    
    log_success "Documentation built successfully"
    
    # Show build size
    BUILD_SIZE=$(du -sh "$BUILD_DIR" | cut -f1)
    log_info "Build size: $BUILD_SIZE"
}

deploy_github_pages() {
    log_info "Deploying to GitHub Pages..."
    
    cd "$DOCS_DIR"
    
    # Check if gh-pages branch exists
    if git show-ref --verify --quiet refs/heads/gh-pages; then
        log_info "gh-pages branch exists"
    else
        log_info "Creating gh-pages branch..."
        git checkout --orphan gh-pages
        git rm -rf .
        git commit --allow-empty -m "Initial gh-pages commit"
        git checkout main
    fi
    
    # Deploy using npm script
    npm run deploy:github
    
    log_success "Deployed to GitHub Pages"
    log_info "Documentation will be available at: https://your-username.github.io/maif/"
}

deploy_netlify() {
    log_info "Preparing for Netlify deployment..."
    
    # Create netlify.toml if it doesn't exist
    if [ ! -f "$DOCS_DIR/netlify.toml" ]; then
        cat > "$DOCS_DIR/netlify.toml" << EOF
[build]
  command = "npm run build"
  publish = ".vitepress/dist"

[build.environment]
  NODE_VERSION = "$NODE_VERSION"

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-XSS-Protection = "1; mode=block"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"

[[headers]]
  for = "*.js"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"

[[headers]]
  for = "*.css"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"
EOF
        log_success "Created netlify.toml"
    fi
    
    log_success "Ready for Netlify deployment"
    log_info "Upload the entire docs/ directory to Netlify or connect your Git repository"
}

deploy_vercel() {
    log_info "Preparing for Vercel deployment..."
    
    # Create vercel.json if it doesn't exist
    if [ ! -f "$DOCS_DIR/vercel.json" ]; then
        cat > "$DOCS_DIR/vercel.json" << EOF
{
  "buildCommand": "npm run build",
  "outputDirectory": ".vitepress/dist",
  "framework": "vitepress",
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        },
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        }
      ]
    }
  ]
}
EOF
        log_success "Created vercel.json"
    fi
    
    log_success "Ready for Vercel deployment"
    log_info "Run 'npx vercel' in the docs/ directory to deploy"
}

create_docker_image() {
    log_info "Creating Docker image..."
    
    # Create Dockerfile if it doesn't exist
    if [ ! -f "$DOCS_DIR/Dockerfile" ]; then
        cat > "$DOCS_DIR/Dockerfile" << 'EOF'
# Multi-stage build for optimized production image
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production --silent

# Copy source code
COPY . .

# Build documentation
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built documentation
COPY --from=builder /app/.vitepress/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Add security headers
RUN echo 'add_header X-Frame-Options "DENY" always;' >> /etc/nginx/conf.d/security.conf && \
    echo 'add_header X-Content-Type-Options "nosniff" always;' >> /etc/nginx/conf.d/security.conf && \
    echo 'add_header X-XSS-Protection "1; mode=block" always;' >> /etc/nginx/conf.d/security.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
EOF
        log_success "Created Dockerfile"
    fi
    
    # Create nginx.conf if it doesn't exist
    if [ ! -f "$DOCS_DIR/nginx.conf" ]; then
        cat > "$DOCS_DIR/nginx.conf" << 'EOF'
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1000;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    # Security headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    server {
        listen 80;
        server_name localhost;
        root /usr/share/nginx/html;
        index index.html;
        
        # Handle client-side routing
        location / {
            try_files $uri $uri/ /index.html;
        }
        
        # Cache static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
        
        # Security
        location ~ /\. {
            deny all;
        }
    }
}
EOF
        log_success "Created nginx.conf"
    fi
    
    # Build Docker image
    docker build -t maif-docs:latest "$DOCS_DIR"
    
    log_success "Docker image created: maif-docs:latest"
    log_info "Run with: docker run -p 8080:80 maif-docs:latest"
}

deploy_kubernetes() {
    log_info "Creating Kubernetes manifests..."
    
    mkdir -p "$DOCS_DIR/k8s"
    
    # Deployment manifest
    cat > "$DOCS_DIR/k8s/deployment.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: maif-docs
  labels:
    app: maif-docs
spec:
  replicas: 3
  selector:
    matchLabels:
      app: maif-docs
  template:
    metadata:
      labels:
        app: maif-docs
    spec:
      containers:
      - name: docs
        image: maif-docs:latest
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: maif-docs-service
spec:
  selector:
    app: maif-docs
  ports:
  - port: 80
    targetPort: 80
  type: LoadBalancer
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: maif-docs-ingress
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - docs.maif.ai
    secretName: maif-docs-tls
  rules:
  - host: docs.maif.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: maif-docs-service
            port:
              number: 80
EOF
    
    log_success "Kubernetes manifests created in k8s/ directory"
    log_info "Deploy with: kubectl apply -f k8s/"
}

run_performance_tests() {
    log_info "Running performance tests..."
    
    cd "$DOCS_DIR"
    
    # Check if Lighthouse CI is available
    if command -v lhci &> /dev/null; then
        log_info "Running Lighthouse CI tests..."
        lhci autorun --upload.target=temporary-public-storage || true
        log_success "Lighthouse tests completed"
    else
        log_warning "Lighthouse CI not installed, skipping performance tests"
        log_info "Install with: npm install -g @lhci/cli"
    fi
    
    # Bundle size analysis
    if [ -d "$BUILD_DIR" ]; then
        log_info "Analyzing bundle size..."
        
        TOTAL_SIZE=$(du -sh "$BUILD_DIR" | cut -f1)
        JS_SIZE=$(find "$BUILD_DIR" -name "*.js" -exec du -ch {} + | grep total | cut -f1)
        CSS_SIZE=$(find "$BUILD_DIR" -name "*.css" -exec du -ch {} + | grep total | cut -f1)
        
        log_info "Bundle Analysis:"
        log_info "  Total size: $TOTAL_SIZE"
        log_info "  JavaScript: $JS_SIZE"
        log_info "  CSS: $CSS_SIZE"
        
        # Check if build size is reasonable
        TOTAL_BYTES=$(du -sb "$BUILD_DIR" | cut -f1)
        if [ "$TOTAL_BYTES" -gt 10485760 ]; then  # 10MB
            log_warning "Build size is larger than 10MB, consider optimization"
        else
            log_success "Build size is optimal"
        fi
    fi
}

generate_sitemap() {
    log_info "Generating sitemap..."
    
    cd "$DOCS_DIR"
    
    if [ -f "scripts/generate-sitemap.js" ]; then
        node scripts/generate-sitemap.js
        log_success "Sitemap generated"
    else
        log_warning "Sitemap generator not found, creating basic script..."
        
        mkdir -p scripts
        cat > "scripts/generate-sitemap.js" << 'EOF'
const fs = require('fs');
const path = require('path');

const baseUrl = 'https://docs.maif.ai';
const buildDir = path.join(__dirname, '..', '.vitepress', 'dist');

function generateSitemap() {
  const urls = [];
  
  function scanDirectory(dir, prefix = '') {
    const files = fs.readdirSync(dir);
    
    for (const file of files) {
      const filePath = path.join(dir, file);
      const stat = fs.statSync(filePath);
      
      if (stat.isDirectory()) {
        scanDirectory(filePath, prefix + '/' + file);
      } else if (file.endsWith('.html')) {
        const url = prefix + '/' + file.replace('.html', '');
        urls.push(baseUrl + url.replace('/index', ''));
      }
    }
  }
  
  scanDirectory(buildDir);
  
  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map(url => `  <url>
    <loc>${url}</loc>
    <lastmod>${new Date().toISOString().split('T')[0]}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>`).join('\n')}
</urlset>`;
  
  fs.writeFileSync(path.join(buildDir, 'sitemap.xml'), sitemap);
  console.log(`Generated sitemap with ${urls.length} URLs`);
}

generateSitemap();
EOF
        
        node scripts/generate-sitemap.js
        log_success "Sitemap generated with basic script"
    fi
}

show_help() {
    echo "MAIF Documentation Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  check        Check prerequisites"
    echo "  install      Install dependencies"
    echo "  build        Build documentation"
    echo "  test         Run quality checks and tests"
    echo "  deploy       Deploy to GitHub Pages"
    echo "  netlify      Prepare for Netlify deployment"
    echo "  vercel       Prepare for Vercel deployment"
    echo "  docker       Create Docker image"
    echo "  k8s          Create Kubernetes manifests"
    echo "  sitemap      Generate sitemap"
    echo "  full         Run complete build and deployment pipeline"
    echo "  help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 full              # Complete build and deployment"
    echo "  $0 build             # Just build the documentation"
    echo "  $0 deploy            # Deploy to GitHub Pages"
    echo "  $0 docker            # Create Docker image"
}

# Main execution
case "${1:-help}" in
    check)
        check_prerequisites
        ;;
    install)
        check_prerequisites
        install_dependencies
        ;;
    build)
        check_prerequisites
        install_dependencies
        build_documentation
        ;;
    test)
        check_prerequisites
        install_dependencies
        run_quality_checks
        ;;
    deploy)
        check_prerequisites
        install_dependencies
        run_quality_checks
        build_documentation
        deploy_github_pages
        ;;
    netlify)
        check_prerequisites
        install_dependencies
        run_quality_checks
        build_documentation
        deploy_netlify
        ;;
    vercel)
        check_prerequisites
        install_dependencies
        run_quality_checks
        build_documentation
        deploy_vercel
        ;;
    docker)
        check_prerequisites
        install_dependencies
        run_quality_checks
        build_documentation
        create_docker_image
        ;;
    k8s)
        deploy_kubernetes
        ;;
    sitemap)
        generate_sitemap
        ;;
    performance)
        run_performance_tests
        ;;
    full)
        log_info "Running complete deployment pipeline..."
        check_prerequisites
        install_dependencies
        run_quality_checks
        build_documentation
        generate_sitemap
        run_performance_tests
        log_success "Build pipeline completed successfully!"
        log_info "Ready for deployment to your chosen platform"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

log_success "Script completed successfully!" 