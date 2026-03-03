# Bash Launchers

## Start Fine-Tuning with nohup

From anywhere:

```bash
bash "/home/ubuntu/MultiModal_Video_Model/scripts/bash/run_video_fine_tuning_nohup.sh"
```

The launcher will:
- run `src/video_fine_tuning.py` in background,
- create a timestamped log in `output/nohup_logs/`,
- print the process PID and log path.

## Monitor Logs

```bash
tail -f /home/ubuntu/MultiModal_Video_Model/output/nohup_logs/video_fine_tuning_YYYYmmdd_HHMMSS.txt
```

Or list latest logs first:

```bash
ls -lt /home/ubuntu/MultiModal_Video_Model/output/nohup_logs
```

## Stop a Run

Use the printed PID:

```bash
kill <PID>
```

Or from a `.pid` file:

```bash
kill "$(cat /home/ubuntu/MultiModal_Video_Model/output/nohup_logs/video_fine_tuning_YYYYmmdd_HHMMSS.pid)"
```
