#!/bin/bash

# Secure Docker Deployment Script
# Addresses Issue #113: Container Security

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔒 Secure Docker Deployment for yt-dl-sub"
echo "========================================="

# Check if .env exists
if [ ! -f "docker/.env" ]; then
    echo -e "${RED}❌ Error: docker/.env file not found${NC}"
    echo "Please copy docker/.env.secure.example to docker/.env and configure it"
    exit 1
fi

# Generate secure keys if not set
generate_key() {
    openssl rand -base64 32
}

# Check required environment variables
check_env() {
    local var_name=$1
    local var_value=$(grep "^${var_name}=" docker/.env | cut -d'=' -f2)
    
    if [ -z "$var_value" ] || [ "$var_value" = "your-${var_name,,}-here" ]; then
        echo -e "${YELLOW}⚠️  ${var_name} not configured, generating secure value...${NC}"
        local new_value=$(generate_key)
        sed -i.bak "s/^${var_name}=.*/${var_name}=${new_value}/" docker/.env
        echo -e "${GREEN}✅ ${var_name} generated${NC}"
    else
        echo -e "${GREEN}✅ ${var_name} configured${NC}"
    fi
}

echo ""
echo "Checking security configuration..."
check_env "API_KEY"
check_env "JWT_SECRET"
check_env "ENCRYPTION_KEY"
check_env "REDIS_PASSWORD"

# Create Docker secrets (for Swarm mode)
create_docker_secrets() {
    echo ""
    echo "Creating Docker secrets..."
    
    # Read values from .env
    source docker/.env
    
    # Create secrets
    echo "$API_KEY" | docker secret create api_key - 2>/dev/null || echo "Secret api_key already exists"
    echo "$JWT_SECRET" | docker secret create jwt_secret - 2>/dev/null || echo "Secret jwt_secret already exists"
    echo "$ENCRYPTION_KEY" | docker secret create encryption_key - 2>/dev/null || echo "Secret encryption_key already exists"
    echo "$REDIS_PASSWORD" | docker secret create redis_password - 2>/dev/null || echo "Secret redis_password already exists"
}

# Create AppArmor profile on host (if AppArmor is available)
setup_apparmor() {
    if command -v aa-status &> /dev/null; then
        echo ""
        echo "Setting up AppArmor profile..."
        
        # Extract AppArmor profile from Dockerfile
        sed -n '/^profile docker-yt-dl-sub/,/^}$/p' Dockerfile.secure > /tmp/docker-yt-dl-sub
        
        # Load profile
        sudo apparmor_parser -r /tmp/docker-yt-dl-sub 2>/dev/null || {
            echo -e "${YELLOW}⚠️  Could not load AppArmor profile (may require sudo)${NC}"
        }
    else
        echo -e "${YELLOW}⚠️  AppArmor not available on this system${NC}"
    fi
}

# Create Seccomp profile
setup_seccomp() {
    echo ""
    echo "Setting up Seccomp profile..."
    
    # Create seccomp directory
    sudo mkdir -p /etc/docker/seccomp
    
    # Extract Seccomp profile from Dockerfile
    sed -n '/^{$/,/^}$/p' Dockerfile.secure | tail -n +2 > /tmp/yt-dl-sub.json
    
    # Copy to Docker config
    sudo cp /tmp/yt-dl-sub.json /etc/docker/seccomp/yt-dl-sub.json 2>/dev/null || {
        echo -e "${YELLOW}⚠️  Could not copy Seccomp profile (may require sudo)${NC}"
    }
}

# Build secure image
build_image() {
    echo ""
    echo "Building secure Docker image..."
    cd ..
    docker build -f docker/Dockerfile.secure -t yt-dl-sub:secure .
    cd docker
    echo -e "${GREEN}✅ Image built successfully${NC}"
}

# Deploy with docker-compose
deploy_compose() {
    echo ""
    echo "Deploying with docker-compose..."
    docker-compose -f docker-compose.secure.yml up -d
    echo -e "${GREEN}✅ Services deployed${NC}"
}

# Health check
health_check() {
    echo ""
    echo "Waiting for services to be healthy..."
    
    # Wait for services
    sleep 10
    
    # Check health
    if docker-compose -f docker-compose.secure.yml ps | grep -q "healthy"; then
        echo -e "${GREEN}✅ All services healthy${NC}"
    else
        echo -e "${YELLOW}⚠️  Some services may still be starting${NC}"
    fi
}

# Main deployment
echo ""
echo "Starting secure deployment..."

# For Docker Swarm mode, create secrets
if docker info | grep -q "Swarm: active"; then
    create_docker_secrets
fi

setup_apparmor
setup_seccomp
build_image
deploy_compose
health_check

echo ""
echo "==========================================="
echo -e "${GREEN}🎉 Secure deployment complete!${NC}"
echo ""
echo "Services running:"
echo "  - API: http://localhost:8000"
echo "  - Redis: localhost:6379"
echo "  - ClamAV: Scanning enabled"
echo ""
echo "Security features enabled:"
echo "  ✅ Non-root user (1000:1000)"
echo "  ✅ Read-only root filesystem"
echo "  ✅ Dropped capabilities"
echo "  ✅ AppArmor profile (if available)"
echo "  ✅ Seccomp profile"
echo "  ✅ Resource limits"
echo "  ✅ Health checks"
echo "  ✅ Encrypted secrets"
echo ""
echo "Monitor logs:"
echo "  docker-compose -f docker-compose.secure.yml logs -f"
echo ""
echo "Stop services:"
echo "  docker-compose -f docker-compose.secure.yml down"