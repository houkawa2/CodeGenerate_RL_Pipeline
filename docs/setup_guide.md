# Environment Setup Guide

## Hardware Requirements

- 4× NVIDIA A800-SXM4-80GB (or equivalent ≥80GB VRAM)
- ≥504GB CPU RAM (for Adam Offload)
- ≥10GB /dev/shm (critical for NCCL)
- ≥200GB disk

## Dependency Version Matrix

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10 | |
| PyTorch | 2.4.0+cu121 | |
| xformers | 0.0.27.post2 | Replaces flash_attn |
| vLLM | 0.6.0 | Imported by OpenRLHF |
| DeepSpeed | 0.16.4 | `--no-deps` install |
| OpenRLHF | commit 2db547e | With GRPO patch |
| Ray | 2.x | Distributed orchestration |

## Common Issues

### NCCL /dev/shm too small
Set shared memory ≥10GB at instance creation. Default 64MB causes NCCL failures.

### flash_attn ABI mismatch
Use xformers instead. Apply flash_attn optional import patch.

### Disk full from checkpoints
DeepSpeed checkpoints include optimizer states (~110GB for 7B). Set `--save_steps` conservatively and clean old checkpoints.
