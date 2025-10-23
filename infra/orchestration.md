# Browser Bot Orchestration

This document explains how to scale and manage multiple browser bot containers for the Clerk AI Reception system.

## Overview

The browser bot orchestration system manages the lifecycle of headless browser containers that join meetings and stream audio. It supports both local development (Docker) and production deployment (AWS ECS/Fargate).

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Scheduler     │───▶│   Bot Manager    │───▶│   Bot Containers│
│   Service       │    │   (Lambda/ECS)   │    │   (Fargate)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   SQS Queue     │    │   CloudWatch     │    │   RT Gateway    │
│   (Bot Jobs)    │    │   (Logs/Metrics) │    │   (WebSocket)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Scaling Strategies

### Horizontal Scaling

Scale bot containers based on meeting demand:

```python
# Auto-scaling configuration
{
    "min_capacity": 0,
    "max_capacity": 50,
    "target_cpu_utilization": 70,
    "scale_out_cooldown": 300,
    "scale_in_cooldown": 600
}
```

### Vertical Scaling

Adjust resource allocation per container:

```yaml
# Resource limits
resources:
  requests:
    cpu: "1"
    memory: "2Gi"
  limits:
    cpu: "2"
    memory: "4Gi"
```

## Deployment Configurations

### Local Development

```bash
# Single bot instance
docker run -d \
  --name clerk-bot-1 \
  --network host \
  --shm-size=2g \
  -e MEETING_URL="https://meet.google.com/test" \
  clerk-browser-bot

# Multiple bot instances
for i in {1..3}; do
  docker run -d \
    --name clerk-bot-$i \
    --network host \
    --shm-size=2g \
    -e MEETING_URL="https://meet.google.com/test-$i" \
    -e SESSION_ID="session-$i" \
    clerk-browser-bot
done
```

### Docker Compose

```yaml
version: '3.8'
services:
  browser-bot-manager:
    image: clerk-bot-manager
    environment:
      - MAX_CONCURRENT_BOTS=10
      - DOCKER_HOST=unix:///var/run/docker.sock
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    depends_on:
      - api
      - rt-gateway
```

### AWS ECS/Fargate

```hcl
resource "aws_ecs_service" "browser_bot_service" {
  name            = "clerk-browser-bot-service"
  cluster         = aws_ecs_cluster.clerk_cluster.id
  task_definition = aws_ecs_task_definition.browser_bot.arn
  desired_count   = 0
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.browser_bot_sg.id]
    assign_public_ip = false
  }
}
```

## Concurrency Management

### Bot Limits

Configure maximum concurrent bots per environment:

```python
# Environment-specific limits
DEVELOPMENT_MAX_BOTS = 5
STAGING_MAX_BOTS = 20
PRODUCTION_MAX_BOTS = 100
```

### Queue Management

Use SQS for bot job queuing:

```python
# Bot job message structure
{
    "meeting_id": "meeting-123",
    "meeting_url": "https://meet.google.com/abc-defg-hij",
    "platform": "google_meet",
    "bot_name": "AI Assistant",
    "priority": "high",
    "retry_count": 0,
    "max_retries": 3
}
```

### Load Balancing

Distribute bot load across multiple availability zones:

```yaml
# ECS service configuration
placement_constraints:
  - type: memberOf
    expression: "attribute:ecs.availability-zone =~ us-east-1*"
```

## Monitoring and Observability

### Metrics

Track key performance indicators:

- Active bot count
- Meeting join success rate
- Audio streaming quality
- Container resource utilization
- Error rates by platform

### Logging

Centralized logging with CloudWatch:

```python
# Structured logging
{
    "timestamp": "2024-01-15T10:30:00Z",
    "level": "INFO",
    "service": "browser-bot",
    "meeting_id": "meeting-123",
    "session_id": "session-456",
    "platform": "google_meet",
    "event": "meeting_joined",
    "duration_ms": 15000
}
```

### Alerting

Set up alerts for:

- Bot failure rate > 5%
- Container memory usage > 80%
- Meeting join timeout > 60s
- WebSocket connection failures

## Resource Optimization

### Memory Management

Optimize Chrome memory usage:

```dockerfile
# Chrome flags for memory optimization
ENV CHROME_FLAGS="--memory-pressure-off --max_old_space_size=4096"
```

### CPU Optimization

Use CPU limits and reservations:

```yaml
resources:
  requests:
    cpu: "0.5"
  limits:
    cpu: "1"
```

### Network Optimization

Optimize network performance:

```yaml
# Use host networking for better performance
network_mode: host

# Or use custom network with optimized settings
networks:
  clerk-network:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.enable_icc: "true"
      com.docker.network.bridge.enable_ip_masquerade: "true"
```

## Security Considerations

### Container Security

- Run containers as non-root user
- Use read-only filesystems where possible
- Scan container images for vulnerabilities
- Implement network policies

### Network Security

- Use private subnets for bot containers
- Implement security groups
- Encrypt traffic between services
- Use VPC endpoints for AWS services

### Access Control

- Implement IAM roles with minimal permissions
- Use secrets manager for sensitive data
- Rotate credentials regularly
- Audit access logs

## Disaster Recovery

### Backup Strategy

- Backup bot configurations
- Store meeting metadata in persistent storage
- Implement cross-region replication

### Failover

- Multi-AZ deployment
- Automatic failover for critical services
- Graceful degradation when resources are limited

## Performance Tuning

### Container Optimization

```dockerfile
# Multi-stage build for smaller images
FROM node:18-slim as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-slim
COPY --from=builder /app/node_modules ./node_modules
COPY . .
```

### Runtime Optimization

```bash
# Optimize Chrome for containerized environment
--no-sandbox
--disable-setuid-sandbox
--disable-dev-shm-usage
--disable-gpu
--disable-web-security
--disable-features=VizDisplayCompositor
```

## Cost Optimization

### Spot Instances

Use spot instances for non-critical workloads:

```hcl
resource "aws_ecs_capacity_provider" "spot" {
  name = "spot"

  auto_scaling_group_provider {
    auto_scaling_group_arn = aws_autoscaling_group.spot.arn
    managed_termination_protection = "DISABLED"
  }
}
```

### Right-sizing

Monitor and adjust resource allocation:

```python
# Dynamic resource allocation based on workload
def calculate_resources(meeting_count, platform):
    base_cpu = 0.5
    base_memory = 1024
    
    if platform == "teams":
        base_cpu += 0.2  # Teams requires more resources
        base_memory += 512
    
    return {
        "cpu": base_cpu * meeting_count,
        "memory": base_memory * meeting_count
    }
```

## Troubleshooting

### Common Issues

1. **Bot containers failing to start**
   - Check resource limits
   - Verify image availability
   - Check security group rules

2. **High memory usage**
   - Monitor Chrome memory leaks
   - Implement container restarts
   - Optimize browser flags

3. **Network connectivity issues**
   - Verify VPC configuration
   - Check security group rules
   - Test DNS resolution

### Debug Commands

```bash
# Check container status
docker ps -a | grep clerk-bot

# View container logs
docker logs clerk-bot-1

# Monitor resource usage
docker stats clerk-bot-1

# Check network connectivity
docker exec clerk-bot-1 ping rt-gateway
```

## Future Enhancements

- Kubernetes operator for advanced orchestration
- Machine learning for predictive scaling
- Multi-cloud deployment support
- Advanced monitoring and alerting
- Automated performance optimization
