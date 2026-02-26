# Docker Guide for MCR

## Quick Commands

### Build and Start

```bat
# Initial build
docker build -t mcr:latest .

# Create and start container
docker run -d --name mcr-container -p 6379:6379 -p 8888:8888 ^
  -v "%cd%\logs:/app/logs" ^
  -v "%cd%\workers:/app/workers" ^
  -v "%cd%\results:/app/results" ^
  -v "%cd%\fig:/app/fig" ^
  mcr:latest
```

### Container Management

```bash
# Start existing container
docker start mcr-container

# Stop container
docker stop mcr-container

# Restart container
docker restart mcr-container

# Remove container
docker rm mcr-container

# Show running containers
docker ps

# Show all containers (including stopped)
docker ps -a
```

### Image Management

```bash
# Remove image
docker rmi mcr:latest

# Full rebuild (no cache)
docker build --no-cache -t mcr:latest .

# List images
docker images
```

### Container Access

```bash
# Interactive shell
docker exec -it mcr-container /bin/bash

# Run single command
docker exec mcr-container python --version

# Interactive command
docker exec -it mcr-container python
```

### Logs and Debugging

```bash
# Show container logs
docker logs mcr-container

# Follow logs in real time
docker logs -f mcr-container

# Last 100 lines
docker logs --tail 100 mcr-container

# Inspect container
docker inspect mcr-container
```

## Volumes and Persistence

### Shared Directories

Volumes let you share data between the host and the container:

| Host Directory | Container Directory | Purpose                         |
| -------------- | ------------------- | ------------------------------- |
| `./logs`       | `/app/logs`         | Worker logs                     |
| `./workers`    | `/app/workers`      | Custom worker scripts           |
| `./results`    | `/app/results`      | Processed results               |
| `./fig`        | `/app/fig`          | Figures and plots               |

### File Access

Files are available from both host and container:

**From the host (Windows):**

```bat
type results\test_results.json
notepad logs\worker.log
```

**From the container:**

```bash
docker exec mcr-container cat /app/results/test_results.json
docker exec mcr-container tail /app/logs/worker.log
```

### Data Backup

To back up data:

```bat
# Copy from container to host
docker cp mcr-container:/app/data ./backup_data

# Copy from host to container
docker cp ./backup_data mcr-container:/app/data
```

## Ports and Services

### Redis (port 6379)

Redis runs inside the container and is reachable from the host:

```bash
# From Windows host (if redis-cli installed)
redis-cli -h localhost -p 6379 ping

# From the container
docker exec mcr-container redis-cli ping
```

### Jupyter (port 8888)

If Jupyter is configured:

```bash
# Start Jupyter inside the container
docker exec -d mcr-container jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Open in host browser
http://localhost:8888
```

## Troubleshooting

### Container Fails to Start

**Issue**: `docker start mcr-container` fails

**Fixes**:

1. Check whether it exists:

   ```bash
   docker ps -a | findstr mcr-container
   ```
2. Remove and recreate:

   ```bat
   run.bat clean-rebuild
   ```
3. Inspect logs:

   ```bash
   docker logs mcr-container
   ```

### Port Already in Use

**Issue**: "port is already allocated" error

**Fixes**:

1. Find the process using the port:

   ```bat
   netstat -ano | findstr :6379
   ```
2. Stop the conflicting service/process
3. Or use different host ports:

   ```bash
   docker run -p 6380:6379 -p 8889:8888 ...
   ```

### Outdated Image

**Issue**: Code changes do not take effect

**Fix**: Rebuild the image

```bat
run.bat rebuild
```

### Low Disk Space

**Issue**: Docker consumes too much space

**Fixes**:

1. Remove unused containers:

   ```bash
   docker container prune
   ```
2. Remove unused images:

   ```bash
   docker image prune
   ```
3. Full cleanup (WARNING: removes everything):

   ```bash
   docker system prune -a
   ```

### Volumes Not Syncing

**Issue**: Changes on the host are not visible in the container

**Fixes**:

1. Check volume mounts:

   ```bash
   docker inspect mcr-container | findstr Mounts -A 20
   ```
2. Restart the container:

   ```bat
   run.bat restart
   ```
3. Recreate the container with correct volumes:

   ```bat
   run.bat clean-rebuild
   ```

### Permission Denied

**Issue**: Permission errors inside the container

**Fix**: Ensure the mounted directories have the right permissions on Windows

## Advanced Configuration

### Custom Dockerfile

To modify the image:

1. Edit `Dockerfile`
2. Rebuild:
   ```bat
   run.bat rebuild
   ```

### Environment Variables

Pass variables to the container:

```bash
docker run -e REDIS_HOST=localhost -e WORKERS=4 mcr:latest
```

Or load them from an `.env` file:

```bash
docker run --env-file .env mcr:latest
```

### Networking

Connect multiple containers:

```bash
# Create network
docker network create mcr-network

# Start containers on that network
docker run --network mcr-network --name mcr-container mcr:latest
docker run --network mcr-network --name redis redis:latest
```

### Resource Limits

Limit container resources:

```bash
docker run --memory="2g" --cpus="2.0" mcr:latest
```

## Useful Commands

### Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Full cleanup
docker system prune -a --volumes
```

### Monitoring

```bash
# Real-time stats
docker stats mcr-container

# Disk usage summary
docker system df

# Processes inside the container
docker top mcr-container
```

### File Copy

```bash
# Container to host
docker cp mcr-container:/app/results/data.json ./data.json

# Host to container
docker cp ./config.yaml mcr-container:/app/config.yaml
```

## References

- [Docker documentation](https://docs.docker.com/)
- [Docker best practices](https://docs.docker.com/develop/dev-best-practices/)
- [Dockerfile reference](https://docs.docker.com/engine/reference/builder/)

## Support

For Docker-related issues:

1. Check logs: `docker logs mcr-container`
2. Inspect the container: `docker inspect mcr-container`
3. Check resource usage: `docker stats mcr-container`
