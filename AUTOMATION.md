# 自动化任务说明（football-predictor）

仓库：https://github.com/zh1592230-cyber/football-predictor

## 1. GitHub Actions（推荐）

工作流文件：`.github/workflows/daily_jingcai.yml`

- **定时**：每天 UTC 00:00 = 北京时间 08:00
- **手动**：Actions → daily-jingcai → Run workflow

### 必配 Secret

| Name | 说明 |
|------|------|
| `FOOTBALL_SCHEDULE_URL` | 赛程 JSON 接口，可含 `{date}` |
| `FOOTBALL_ODDS_URL` | 可选，赔率接口，可含 `{match_id}` |
| `FOOTBALL_API_TOKEN` | 可选，作为 Authorization 头 |

路径：Settings → Secrets and variables → Actions → New repository secret

### 配置后验证

1. Actions → **daily-jingcai** → **Run workflow**
2. 成功后下载 Artifact `jingcai-YYYY-MM-DD`
3. 查看 Summary 与 `data/logs/`

未配置赛程 URL 时，可勾选 **demo**（需仓库内有对应 `data/samples`）。

## 2. 本地 cron

```bash
chmod +x run_daily.sh
# .env 中写入 FOOTBALL_SCHEDULE_URL 等
./run_daily.sh
./run_daily.sh 2026-11-08

# crontab（北京 08:00）
# 0 8 * * * TZ=Asia/Shanghai /bin/bash -lc 'cd /path/to/football-predictor && ./run_daily.sh'
```

## 3. 入口命令

```bash
python scripts/run_today.py
python scripts/run_today.py --date 2026-11-08 --recommend
python scripts/run_today.py --test-schedule
```
