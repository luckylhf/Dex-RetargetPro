# Robot-Retarget 1.0.4

将 BVH 或 SMPL-X 人体动作转换为 Dex 机器人的 50 Hz、29 轴轨迹。用 `fast` 预览动作，用 `refine` 转换并验收训练轨迹。两种模式的默认精修路线均为 A。

## 安装和运行

推荐 Python 3.12。在解压后的源码包或 Git 工作区根目录执行。安装从 Python 包索引获取求解依赖。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -c requirements-lock.txt .
export DEX_FRAX_ASSETS="$PWD/assets/dex_evt"
```

从 wheel 安装时，将安装命令中的 `.` 替换成 wheel 路径，并将 `DEX_FRAX_ASSETS` 指向同版本源码包中的 `assets/dex_evt`。约束文件也在源码包根目录。使用 LaFAN 或 SOMA 输入时，按来源说明准备对应的参考几何，并设置 `DEX_FRAX_SOURCE_GEOMETRY`。 [机器人资产](assets/dex_evt/README.md) · [源几何和模型](docs/INPUTS.md)

## 转换 BVH

先用一小段确认源骨架、单位和位置通道约定。示例使用 `mocap` 骨架、厘米和绝对位置通道。请按实际输入选择 `mocap`、`lafan` 或 `soma`。每次转换使用新输出目录。 [BVH 选项](docs/INPUTS.md#bvh)

```sh
dex-frax convert /path/to/input.bvh outputs/bvh_preview \
  --source mocap --mode fast --bvh-units cm --bvh-translations absolute \
  --max-frames 60 --repair-budget 20

dex-frax convert /path/to/input.bvh outputs/bvh_refine \
  --source mocap --mode refine --bvh-units cm --bvh-translations absolute
```

回放优先使用 `frax/motion.npz`。验收状态和未解决事项见 `frax/pipeline.md` 与 `frax/quality.json`。`review` 轨迹供预览和排查。`refine` 通过终验后另写 `frax/training.npz`。预览和续跑工具只应读取可信来源的机器人 PKL。 [输出与排查](docs/VALIDATION.md)

## 转换 SMPL-X

如使用同一 Python 环境，在已安装转换器的环境中执行以下命令。权重按 SMPL-X 许可获取，并放在权重父目录的 `smplx/` 下。`--input-fps` 按源数据实际帧率填写。

```sh
python -m pip install "torch>=2.6" "smplx==0.1.28"
dex-frax convert /path/to/input.pt outputs/smplx_refine \
  --source smplx --mode refine --input-fps 30 --grounding contact \
  --body-models /path/to/body_models
```

如已有独立的 SMPL-X 环境，可通过 `--smplx-python` 指定其解释器。 [SMPL-X 选项](docs/INPUTS.md#smpl-x)

## 精修路线和预算

### 精修流程 A、B、C

`convert` 和 `batch` 省略 `--refine-flow` 时均走 A。测试 B、C 时，用 `--mode refine --refine-flow B` 或 `--mode refine --refine-flow C`，分别写入新目录。

| 路线 | 精修调度 |
|---|---|
| **A · 默认基线** | 按专项顺序轮转处理腿部避碰、脚部支撑、穿地、局部去抖、身体支撑和手臂避碰。 |
| **B · 测试线** | 先处理优先落地窗口，再交替派发专项和 V8 连续段，最后用余量做时序修复。 |
| **C · 测试线** | 按支撑相位联解同窗的脚腿净空、碰撞和身体支撑，按残差及专项进展派发 V8。 |

同一模式的三条路线共用输入审查、Frax 初解、保护约束和终验。 [完整流程](docs/PIPELINE.md)

默认精修时限为原始整条动作换算到 50 Hz 的总帧数 × 0.5 秒。`--repair-budget` 可指定秒数，`--repair-budget unlimited` 可取消共享时限。单次求解上限仍为专项 20 秒及 B、C 的 V8 单段 50 秒。`--budget` 限制整次转换或整个批次，默认分别为 24 小时和 1,800 秒。 [批处理与预算](docs/BATCH.md)

### 精修预算用完后续跑

当 `frax/pipeline.md` 的「精修状态」显示共享时限耗尽且还有任务，在同一构建下保留源文件、模型和转换选项，用 `--continue-from` 指向报告中的检查点，用 `--repair-budget` 增加本轮时限。B 或 C 的检查点须显式指定相同路线。

```sh
dex-frax convert /path/to/input.bvh outputs/bvh_round2 \
  --source mocap --mode refine --bvh-units cm --bvh-translations absolute \
  --continue-from outputs/bvh_refine/frax --repair-budget 900
```

如果报告将检查点标为 `pre_final_retime`，使用 `outputs/bvh_refine/frax/pre_final_retime`。 [读报告与处理残余问题](docs/VALIDATION.md)

## 参考

- [当前版本和适用范围](docs/CURRENT_VERSION.md)
- [输入、接触与输出字段](docs/INPUTS.md)
- [处理流程](docs/PIPELINE.md) · [编号流程图](docs/REFINE_FLOWCHART.html)
- [候选采用与验收规则](docs/SAFETY_REPAIR_RULES.md)
- [批处理与续跑](docs/BATCH.md)
- [第三方来源与许可](THIRD_PARTY.md)
