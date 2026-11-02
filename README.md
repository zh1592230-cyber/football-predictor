# Football Predictor v0.9.41

每日竞彩足球赛前分析工具。当前仓库提供可配置的数据适配器和明确标注的基线模型；**不内置赛程或 NowGoal 数据，也不代表已接入任何真实供应商**。未配置数据源时，实时入口会报出配置错误，不会把固定样例伪装成当日数据。

## 能力与边界

- 通过 `FOOTBALL_SCHEDULE_URL` 拉取 JSON 赛程；可按日期缓存并报告数据来源、更新时间和缓存状态。
- 通过 `FOOTBALL_ODDS_URL` 拉取 JSON 赔率；接口是通用适配器，并非 NowGoal 官方 API。请仅接入有权使用的稳定数据源。
- 基于输入 xG（缺失时显式使用配置中的未校准基线；提供 xG 时按 `lambda_shrink` 向该基线收缩）计算 Poisson + Dixon–Coles 得分矩阵。
- 可在 1X2 概率层融合去水市场概率；不从 1X2 赔率反推进球均值。
- 结算预测与最终比分，计算 1X2 Brier 分数和 log loss；有足够的带标签样本时，可按时间切分扫描概率融合权重。
- 不支持滚球条件预测。标记为 `live` 的比赛会跳过，避免将赛前模型误用于比赛进行中。

提供方应返回赛程 JSON 数组，或 `{"matches": [...]}`。每场至少包含稳定 `match_id`、`home_team`、`away_team`、`status` 和带时区的 ISO-8601 `kickoff`；`status` 应为 `scheduled`、`live`、`finished`、`postponed` 或 `cancelled`，常见 `not_started`、`in_play`、`inplay`、`ft` 别名会归一化。空赛程、重复 ID、无效时间或只给一方 xG 都会报错，不解释成“确认无赛事”。可选联赛字段为 `league`。xG 的规范输入是嵌套对象 `prematch_xg: {"home": 1.4, "away": 0.9, "type": "prematch", "source": "licensed-provider-id", "generated_at": "2026-11-03T08:00:00Z"}`；扁平 `home_xg`/`away_xg` 仅作尚未核验来源的兼容输入，不能触发推荐。provider xG 仅在 `source` 位于配置 `xg_allowed_sources`、生成时间早于预测时间和开赛时间、且未超过 `xg_ttl_seconds` 时被视为可验证赛前数据。赛后实际 xG (`xg_type: postmatch_actual_xg`) 不可进入预测输入。赔率 JSON 对象应包含大于 1 的 `home`、`draw`、`away` 十进制赔率，以及带时区的 ISO-8601 `updated_at` 时间戳。需要认证时，可设置 `FOOTBALL_API_TOKEN` 环境变量；适配器会将其作为 HTTP Authorization 请求头发送。请勿提交密钥。

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
python scripts/run_today.py --test-schedule
python scripts/run_today.py --test-schedule --no-cache
```

若 URL 含 `{date}`，请求时会替换为 `YYYY-MM-DD`；否则适配器会添加 `date` 查询参数。赔率 URL 可含 `{match_id}`，或由适配器添加 `match_id` 查询参数。两种 URL 都可用 `FOOTBALL_PREDICTOR_DEMO=1` 启用本地 `data/samples/schedule_YYYY-MM-DD.json` 样例模式；该模式不请求供应商，且没有内置样例数据。样例结果会标记为 `demo`，不得用于实时决策。

`--no-cache` 跳过赛程缓存读取并请求配置的数据源（成功后仍更新缓存）。`--test-schedule` 只检查赛程并打印来源、更新时间、新鲜度和场次数；获取错误返回退出码 2，过期缓存、demo fixture 或零场返回非零退出码。Demo fixture 会明确报告 `source=demo_fixture` / `freshness=demo`，不是实时源连通性证明。未配置 `FOOTBALL_SCHEDULE_URL` 且无可用缓存时，今日预测和赛程检查均明确失败，不会伪称成功。没有 `FOOTBALL_ODDS_URL` 时预测会标记赔率未配置并仅输出模型概率，不含赔率融合。

常用配置位于 `configs/params_v0941.json`。其中 0.9.41 参数和联赛 `home_boost` 是用户提供的初始参考值，未声称经过本项目回测验证；当前模型仅使用明确说明的 Poisson/DC 参数、基线 xG、最大进球范围和融合权重。`lambda_shrink` 是 match xG 向配置基线的收缩权重，联赛 `home_boost` 只调整该主场基线，不代表模型已校准。预测结果会在 `xg_source` 和 `expected_goals.metadata` 标记来源、时间和新鲜度。推荐只对配置允许且新鲜、开赛前生成的 provider xG 开启，并仍需达到置信度阈值；基线/未验证 xG 不产生推荐，`allow_baseline_recommend` 不会绕过这一数据来源门槛。用户提供的演示赛程/比分样例仅用于契约测试，未写入真实赛程数据，也未用于训练或参数扫描。历史结果只用于结算；生成历史赛前预测时，特征必须在该场 kickoff 前可用，不能把赛后比分或实际 xG 输入预测模型。

## 回测与权重扫描

结算输入 JSON 格式为 `[{"match_id":"...","home_goals":1,"away_goals":0}]`。预测文件应是每日脚本生成的 JSON 数组。结算报告保留模型/市场概率，可在有足够样本及赔率记录时直接用于权重扫描：

```powershell
python scripts/run_backtest.py --predictions data/daily/predictions_2026-11-01.json --results data/samples/results.json
python scripts/scan_params.py --input data/samples/labeled_predictions.json
```

扫描文件中的每条历史记录必须包含 `match_id`、`date`、`model_probabilities`、`market_probabilities` 和 `actual_outcome`（`home`、`draw` 或 `away`）；可以是该记录数组，也可以是结算报告的 `matches` 数组。扫描按日期排序，使用前 70% 选择融合权重、后 30% 验证；训练或验证样本不足时明确报错，不修改参数文件。得分适用于所提供的带标签样本，不等同于前瞻性能保证。

`src/data/schedule.py` 和 `src/data/nowgoal.py` 说明了供应商响应契约；没有凭据或稳定供应商时请勿将适配器名称理解为官方集成。当前不抓取网站、不绕过访问控制，也不提供滚球比分、分钟或红牌条件模型。
