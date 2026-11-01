# Football Predictor v0.9.41

每日竞彩足球赛前分析工具。当前仓库提供可配置的数据适配器和明确标注的基线模型；**不内置赛程或 NowGoal 数据，也不代表已接入任何真实供应商**。未配置数据源时，实时入口会报出配置错误，不会把固定样例伪装成当日数据。

## 能力与边界

- 通过 `FOOTBALL_SCHEDULE_URL` 拉取 JSON 赛程；可按日期缓存并报告数据来源、更新时间和缓存状态。
- 通过 `FOOTBALL_ODDS_URL` 拉取 JSON 赔率；接口是通用适配器，并非 NowGoal 官方 API。请仅接入有权使用的稳定数据源。
- 基于输入 xG（缺失时显式使用配置中的未校准基线；提供 xG 时按 `lambda_shrink` 向该基线收缩）计算 Poisson + Dixon–Coles 得分矩阵。
- 可在 1X2 概率层融合去水市场概率；不从 1X2 赔率反推进球均值。
- 结算预测与最终比分，计算 1X2 Brier 分数和 log loss；有足够的带标签样本时，可按时间切分扫描概率融合权重。
- 不支持滚球条件预测。标记为 `live` 的比赛会跳过，避免将赛前模型误用于比赛进行中。

提供方应返回赛程 JSON 数组，或 `{"matches": [...]}`。每场至少包含 `match_id`、`home_team`、`away_team`、`status`；`status` 应为 `scheduled`、`live`、`finished`、`postponed` 或 `cancelled`。可选字段包括带时区的 ISO-8601 `kickoff`、`league`、`home_xg`、`away_xg`。赔率 JSON 对象应包含大于 1 的 `home`、`draw`、`away` 十进制赔率，以及带时区的 ISO-8601 `updated_at` 时间戳。需要认证时，可设置 `FOOTBALL_API_TOKEN` 环境变量；适配器会将其作为 HTTP Authorization 请求头发送。请勿提交密钥。

Dixon–Coles 使用此处的 rho 符号约定：`tau00=1-lambda_home*lambda_away*rho`、`tau01=1+lambda_home*rho`、`tau10=1+lambda_away*rho`、`tau11=1-rho`，仅校正 0-0、0-1、1-0、1-1。非法负校正会报错。得分矩阵仅计算 0 至 `max_goals`，并在该截断矩阵内归一化；`truncated_score_mass` 报告独立 Poisson 截断前的遗漏尾部质量。

## 快速开始

Python 3.10+，无必需的第三方运行依赖：

```powershell
pip install -r requirements.txt
```

配置环境变量 `FOOTBALL_SCHEDULE_URL` 后运行：

```powershell
python scripts/run_today.py
python scripts/run_today.py --recommend
python scripts/run_today.py --date 2026-11-01 --leagues PL,PD
```

若 URL 含 `{date}`，请求时会替换为 `YYYY-MM-DD`；否则适配器会添加 `date` 查询参数。赔率 URL 可含 `{match_id}`，或由适配器添加 `match_id` 查询参数。两种 URL 都可用 `FOOTBALL_PREDICTOR_DEMO=1` 启用本地 `data/samples/schedule_YYYY-MM-DD.json` 样例模式；该模式不请求供应商，且没有内置样例数据。样例结果会标记为 `demo`，不得用于实时决策。

常用配置位于 `configs/params_v0941.json`。其中 0.9.41 参数和联赛 `home_boost` 是用户提供的初始参考值，未声称经过本项目回测验证；当前模型仅使用明确说明的 Poisson/DC 参数、基线 xG、最大进球范围和融合权重。`lambda_shrink` 是 match xG 向配置基线的收缩权重。经验后处理参数目前保留以便兼容，但不会影响输出。推荐筛选只对有明确 match xG 输入且赛程非 demo/非过期的比赛启用；`--recommend` 不代表投注建议。

## 回测与权重扫描

结算输入 JSON 格式为 `[{"match_id":"...","home_goals":1,"away_goals":0}]`。预测文件应是每日脚本生成的 JSON 数组。结算报告保留模型/市场概率，可在有足够样本及赔率记录时直接用于权重扫描：

```powershell
python scripts/run_backtest.py --predictions data/daily/predictions_2026-11-01.json --results data/samples/results.json
python scripts/scan_params.py --input data/samples/labeled_predictions.json
```

扫描文件中的每条历史记录必须包含 `match_id`、`date`、`model_probabilities`、`market_probabilities` 和 `actual_outcome`（`home`、`draw` 或 `away`）；可以是该记录数组，也可以是结算报告的 `matches` 数组。扫描按日期排序，使用前 70% 选择融合权重、后 30% 验证；训练或验证样本不足时明确报错，不修改参数文件。得分适用于所提供的带标签样本，不等同于前瞻性能保证。

`src/data/schedule.py` 和 `src/data/nowgoal.py` 说明了供应商响应契约；没有凭据或稳定供应商时请勿将适配器名称理解为官方集成。当前不抓取网站、不绕过访问控制，也不提供滚球比分、分钟或红牌条件模型。
