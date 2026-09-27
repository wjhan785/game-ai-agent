[English](README.md) | 简体中文

# game-ai-agent

一个无界面的回合制战术战斗引擎，其中刻意预埋了 11 个缺陷，供一个 LLM
Agent 来搜寻。这个 Agent 会规划下一步测试什么，把记忆保存在上下文窗口之外，
并通过调用工具来行动，而不是对整份状态数据进行自由推理。

六个场景的设计思路见 [docs/scenario-matrix.md](docs/scenario-matrix.md)，
十一个预埋缺陷见 [ANSWER_KEY.md](ANSWER_KEY.md)（自动生成）。

## 安装

```bash
python -m pip install pydantic openai pytest python-dotenv jinja2 numpy
python -m pip install torch        # 可选：仅敌方 AI 需要（enemy_ai/、scripts/play.py）
bash scripts/install-hooks.sh      # 安装 pre-commit 钩子，阻止提交 API 密钥
```

然后将 `.env.example` 复制为 `.env`（已被 gitignore 忽略），并填写
`DEEPSEEK_API_KEY=`。也可以填入你自己的 API。

## 运行

```bash
python -m pytest -q                  # 完整测试套件：离线运行，不调用 API，几秒完成
python -m agent.llm --smoke          # 一次真实的工具调用往返：检查密钥、模型和费用统计
python -m eval.pilot --run-name pilot --episodes 25 --max-spend 0.75   # 完整 Agent 的一轮测试战役 + 缺陷报告
python -m eval.pilot --run-name greedy --method greedy_llm --episodes 25 # Greedy-LLM 基线
python -m eval.random_baseline --run-name random --actions 800          # 随机基线（不调用 API）
python -m eval.bug_report logs/runs/pilot                              # 重新评分一轮已完成的战役
python -m replay.generate logs/runs/pilot/episodes/ep_0001.jsonl out.html
python scripts/play.py S1                                              # 亲自对战敌方 AI
python -m enemy_ai train                                               # 重新训练敌方 AI（单 CPU 线程约 200 秒）
python -m enemy_ai check --greedy                                      # 敌方 AI 对随机策略的胜率
```

所有可能产生费用的入口都接受 `--max-spend` 参数（本次运行的花费上限）。因达到
上限而停止的战役，可以用同一个运行名加 `--resume` 继续。战役默认会把每一对
请求/响应记录到 `fixtures/`（`--mode record`）；`--mode replay` 可离线零成本
重放。

## 架构

```
engine/         游戏本体：纯函数式、确定性、可序列化
  models.py       Pydantic 状态：Character、Ability、StatusEffect、BattleState、Action
  content.py      技能与角色库（纯数据）
  elements.py     火 > 冰 > 雷 > 火，x1.5 / x1.0 / x1/1.5
  effects.py      状态施加、叠加规则、每回合结算逻辑——大部分缺陷在这里
  resolution.py   每个行动轮次的结算流水线（步骤固定，见其 docstring），以及行动值调度器的推进
                  （下一个行动的角色由速度决定，而非固定顺序）
  rules.py        合法行动检查：能量、冷却、目标是否有效
  defects.py      十一个预埋缺陷，以开关形式存在（默认全部为 False）
  invariants.py   每次行动后的不变量检查，与缺陷开关无关
  scenarios.py    六个手工设计的场景（每个角色有各自的速度），通过种子产生有限的变化
  engine.py       BattleEngine 门面：reset / legal_actions / take_action / pending_record / log
  episode_log.py  单局 JSONL：头部（运行元数据 + 初始状态）、每个回合一行、尾部

agent/          LLM Agent——不得导入 engine.defects 或 oracle/（由测试强制保证）
  llm.py          通过 OpenAI SDK 调用 DeepSeek：强制单次工具调用、校验/重试、考虑缓存的
                  费用统计、花费上限、非高峰时段限制、cassette 录制/重放
  prompts.py      模型看到的内容：规则说明、阵容、精简的每回合观察
  tools.py        两层循环使用的工具接口，以及包装单个 BattleEngine 的分发器
  ledger.py       SQLite 外部记忆：对局、假设、标记、访问记录、覆盖率
  inner_loop.py   朝着本局目标做每回合的战术决策；标记异常
  runner.py       单局运行：LLM 调用门控、脚本化回合、台账记录
  outer_loop.py   每局的规划器，以及战役驱动器

baselines/      对比方法——与 agent/ 遵守同样的防泄露规则
  random_explorer.py  双方都均匀随机选择合法行动，不使用 LLM
  greedy_llm.py       关闭规划器和台账回读的完整 Agent

enemy_ai/       游戏自带的敌方 AI，供人类游玩——不参与缺陷测试
  env.py          固定形状的观察/动作编码，以及自我对弈训练环境
  policy.py       PyTorch actor-critic、PPO 自我对弈训练、对随机策略的强度检验
  enemy_policy.pt 训练好的模型检查点

oracle/         仅用于评分——Agent 永远看不到
  differential.py 在另一个构建下重放一条轨迹；找出第一个分歧点
  attribute.py    单开关消融：哪个缺陷能解释一条单缺陷轨迹
  triggers.py     针对全缺陷构建轨迹的留一法触发判定与单步活跃度

eval/
  pilot.py        构建含缺陷的版本（所有预埋缺陷全开），并对其运行 LLM 战役
                  （完整 Agent 或 Greedy-LLM）
  random_baseline.py  同上，用于随机基线
  bug_report.py   为一轮战役评分：每个缺陷的触发与检出情况，并对每个标记做分诊

scripts/
  play.py         通过 Agent 自己的工具接口手动进行一场战斗；另一方由敌方 AI 操控
                  （--manual-enemy 可同时操控双方）

replay/
  generate.py     单局 JSONL -> 自包含的 HTML 回放（Jinja2）
```

**状态简单且可序列化：** `BattleEngine.state_dict()` / `log_dicts()` 通过
Pydantic 把整场战斗（或其逐回合历史）导出为 JSON。`engine/` 中除了
`BattleEngine` 对象本身（它内部记录当前待决策的行动轮次）之外，没有任何不可
序列化的状态。

**行动顺序由速度决定：** 每个角色有一个 `speed` 属性和一个行动值，
AV = 10000 / speed；行动值最低的角色先行动，因此在同样长的战斗时间里，速度快的
角色比速度慢的角色行动更频繁。平局时按各场景声明的顺序（`p1, e1, p2, e2, p3, e3`）
决定。经过 100 AV 的时钟为一个周期。`BattleState.turn_forecast()` 向 Agent
（以及 `scripts/play.py`）提供之后的行动顺序预告，使多角色之间的配合可以被
有意设计，而不是靠运气。

**三种方法：** 完整 Agent（规划器 + 台账 + 每回合循环）、**Greedy-LLM**（相同的
每回合循环、提示词、工具和门控，但关闭规划器和台账回读），以及一个**随机**探索者。
三者都通过同一个引擎操控双方队伍，写入相同的台账和单局日志，并由同一个判定器
（oracle）和缺陷报告评分，因此它们之间的任何差异都来自方法本身。

**供人类游玩的敌方 AI：** `enemy_ai/` 是一个用 PyTorch 在无缺陷构建上训练的小型
PPO 策略，因此它学到的是设计意图中的游戏，而不是预埋的缺陷。它在
`scripts/play.py` 中操控敌方队伍。它不参与缺陷测试；在缺陷测试中，被测方法
同时操控双方队伍。

**Agent 永远不需要提交空操作：** `BattleEngine` 会自动结算不需要真正决策的行动
轮次（已阵亡角色的回合、被眩晕角色的回合、没有任何合法行动的敌人——见缺陷 B08），
只在确实需要做选择时才停下来请求行动。

## 十一个预埋缺陷

这些缺陷由 `engine/defects.py` 中的布尔开关控制（默认全部为 `False`），而不是
写死在引擎唯一的代码路径里。正因如此，`tests/test_engine_golden.py` 可以针对
无缺陷模式断言*正确*的语义，而不必把缺陷编码进测试；
`scripts/generate_answer_key.py` 也能直接从开关自身的 docstring 重新生成
[ANSWER_KEY.md](ANSWER_KEY.md)，而不是维护一份会与代码逐渐脱节的手写文档。
第十一个缺陷 B11 在新的速度调度器中预埋了错误的平局裁决（按角色 id 排序，
而不是按场景声明的顺序），其评分方式与其他缺陷相同：`oracle/triggers.py` 把它
和 B08 一样视为构建时缺陷，因为它在战斗构建时就已确定，而不是在某一步中发生。
十一个缺陷中有六个能单独被不变量检查器发现；另外五个不能。哪些属于哪一类见
ANSWER_KEY.md；为什么这五个缺陷的差距才是本项目真正值得衡量的部分，见下方结果。

## 结果

每种方法各运行一轮试点战役，全部针对全缺陷构建（所有缺陷同时开启），使用
`deepseek-flash`。如果判定器的留一法重放显示某个缺陷改变了某局的轨迹，则该缺陷
在这一局中计为**已触发**；如果不变量检查器报告了它，或者某个 Agent 标记与它强匹配
（标记中点名了受该缺陷影响的角色，且措辞符合该缺陷），则计为**已检出**。

|                                   | 随机      | Greedy-LLM   | 完整 Agent   |
| --------------------------------- | --------- | ------------ | ------------ |
| 对局数 / 行动数                   | 21 / 844  | 25 / 751     | 26 / 871     |
| 模型花费                          | $0        | $0.34        | $0.96        |
| 已触发缺陷                        | 11 / 11   | 11 / 11      | 11 / 11      |
| 已检出缺陷                        | 5 / 11    | 11 / 11      | 11 / 11      |
| 已检出的不变量不可见缺陷          | **0 / 5** | **5 / 5**    | **5 / 5**    |
| Agent 标记（强 / 弱 / 未匹配）    | --        | 200 / 2 / 45 | 186 / 3 / 68 |
| LLM 调用次数（规划器 / 每回合）   | --        | 0 / 709      | 246 / 861    |
| 提示词缓存命中率                  | --        | 68.3%        | 81.0%        |

按缺陷统计，已检出的对局数 / 已触发的对局数：

| 缺陷                                   | 不变量可见 | 随机    | Greedy-LLM | 完整 Agent |
| -------------------------------------- | ---------- | ------- | ---------- | ---------- |
| B01 护盾吸收中毒/灼烧的持续伤害        | 否         | 0 / 1   | 2 / 4      | 3 / 5      |
| B02 灼烧叠层相乘而非相加               | 否         | 0 / 5   | 7 / 8      | 5 / 6      |
| B03 冰冻使能量变为负数                 | 是         | 0 / 4   | 4 / 4      | 3 / 4      |
| B04 中毒按虚弱调整后的生命值计算       | 否         | 0 / 3   | 3 / 4      | 3 / 3      |
| B05 眩晕重复叠加而非刷新               | 是         | 2 / 2   | 1 / 1      | 5 / 5      |
| B06 冷却每回合被扣减两次               | 是         | 13 / 21 | 15 / 25    | 19 / 25    |
| B07 再生效果在死亡后仍然保留           | 是         | 4 / 4   | 8 / 8      | 6 / 6      |
| B08 敌人没有普通攻击作为兜底           | 是         | 2 / 3   | 3 / 4      | 6 / 7      |
| B09 元素倍率被应用两次                 | 否         | 0 / 21  | 25 / 25    | 24 / 25    |
| B10 护盾值归零后未被移除               | 是         | 5 / 5   | 15 / 15    | 10 / 10    |
| B11 行动顺序平局按 id 裁决             | 否         | 0 / 21  | 1 / 24     | 5 / 25     |

**随机游玩能触及每一个缺陷，却漏掉了其中六个。** 它触发了全部十一个缺陷，但由于
只有不变量检查器来报告，那五个产生错误数值（而非不可能状态）的缺陷一个都没有检出
（B03 也没有，见下文）。两种 LLM 方法都检出了全部五个，方法是把每回合观察到的
数值与规则和技能声明的数据进行核对。

**规划器带来的变化。** 完整 Agent 的规划器会选择场景并设置特定的交互，例如行动
顺序平局：B11 在完整 Agent 中有 5 局被匹配，而 Greedy-LLM 只有 1 局。规划器占了
完整 Agent 大部分的花费（26 局中共 246 次规划器调用）。

完整报告（包含每个标记以及每局的回放链接）：
`results/{random,greedy_llm,full_agent}/bug_report.md`（在本地生成；
`results/` 已被 gitignore 忽略）。

## 方法论保障

- **`docs/scenario-matrix.md` 的提交早于 `engine/defects.py` 的存在**，
  `tests/test_no_bug_leakage.py` 会对照 git 历史检查这一先后顺序，因此场景不可能
  是根据答案反推设计出来的。
- **同一个测试文件**还会静态检查 `agent/`、`baselines/` 和 `enemy_ai/` 下的任何
  代码都没有导入 `engine.defects` 或 `oracle/`，并且源码中任何位置（包括字符串
  字面量，因此也覆盖提示词模板）都不包含任何缺陷编号或 `DefectFlags` 字段名。
- **`tests/test_no_secrets.py`** 会扫描工作区中任何形似 API 密钥的内容，并且
  （在设置了 `DEEPSEEK_API_KEY` 时）扫描真实密钥的字面值。
  `scripts/install-hooks.sh` 会安装一个对暂存区差异做同样检查的 pre-commit 钩子。

## 路线图

- ~~R1~~——差分判定器、单局 JSONL 日志、回放 HTML 生成器、LLM 提供方层 + 冒烟
  测试、工具接口、一个最简的单回合 Agent。
- ~~R2~~——探索台账（SQLite）、外层规划循环、接入不变量检查器的
  `flag_anomaly`、第一份缺陷报告。
- ~~R3~~——基线（随机，以及 Greedy-LLM：关闭台账和规划器的同一个 Agent）、三种
  方法的试点运行。

在 Claude Code 的协助下开发。
