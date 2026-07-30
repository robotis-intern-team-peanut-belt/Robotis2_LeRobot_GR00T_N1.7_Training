# OMX Insert weekend training queue

Archived reference only. This queue predates `grootctl`; use the YAML registry
under `configs/` for current work. The scripts retain their historical paths
and are not supported launch entry points after repository consolidation.

`weekend_queue.tsv` is the ordered experiment definition. All runs use LeRobot
GR00T N1.7, batch 8, 80,000 optimizer steps, horizon 40, a 10% final-episode
holdout, W&B project `gr00t-omx-insert`, and no W&B model upload.

Start:

```bash
./start_weekend_queue.sh
```

Monitor:

```bash
tail -f ../runs/2026-07-24_weekend_queue/queue_supervisor.log
column -ts $'\t' ../runs/2026-07-24_weekend_queue/queue_status.tsv
```

Stop the queue and its active trainer:

```bash
./stop_weekend_queue.sh
```

Failures are recorded and the next run continues. CUDA OOM is explicitly
classified. Existing outputs are never overwritten, duplicate queue processes
are rejected with a lock, and the queue stops before starting another run if
free space falls below 100 GiB. A successful run receives `QUEUE_SUCCESS`;
rerunning the queue skips it.
