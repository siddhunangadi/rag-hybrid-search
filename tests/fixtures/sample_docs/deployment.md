# Deployment Guide

## Common Errors

If you see ERROR_CODE_0x834 during deployment, it means the persistent
disk was not mounted before the service started. Remount the disk and
restart the container.

## Rollback Procedure

To roll back a bad deployment, redeploy the previous Docker image tag and
verify health checks pass before routing traffic to it.
