# Dex-RetargetPro

将 BVH 或 SMPL-X 人体动作转换为 Dex 机器人的 50 Hz、31 轴轨迹。用 `fast` 预览动作，用 `refine` 转换并验收训练轨迹。

## 安装和运行

推荐 Python 3.12。在解压后的源码包或 Git 工作区根目录执行。安装从 Python 包索引获取求解依赖。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -c requirements-lock.txt .
export DEX_FRAX_ASSETS="$PWD/assets/dex_evt"
```

从 wheel 安装时，将安装命令中的 `.` 替换成 wheel 路径，并将 `DEX_FRAX_ASSETS` 指向配套的 Dex 31 轴资产目录。约束文件位于源码包根目录。LaFAN 和 SOMA 输入还需配置人体表面几何。 [机器人资产](assets/dex_evt/README.md) · [输入配置](docs/INPUTS.md)

## 转换 BVH

根据输入文件选择骨架、单位和位置通道约定。下列命令使用 `mocap` 骨架、厘米和绝对位置通道。每次转换使用新输出目录。 [BVH 选项](docs/INPUTS.md#bvh)

```sh
dex-frax convert /path/to/input.bvh outputs/bvh_preview \
  --source mocap --mode fast --bvh-units cm --bvh-translations absolute \
  --max-frames 60

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

## 头部动作

头部 yaw、pitch 跟踪人体头部相对躯干的朝向，导出关节名末两列为 `head_yaw_joint` 和 `head_pitch_joint`。源目标超出可达角度时，在机器人限位内尽量跟踪，触限饱和不报错。BVH 缺失头部关节时，头部目标取躯干相对零位。跟踪差异和校准来源见 `frax/quality.json` 的 `head_tracking`。 [头部硬限位规则](docs/SAFETY_REPAIR_RULES.md#头部硬限位)

## 脚部几何与碰撞检查

离线转换用脚部视觉网格的整脚凸包确定落脚高度，检查脚地穿透和脚与脚或腿的自碰。提供 `--scene` 时，使用实测盒体检查平台接触与碰撞。LAFAN 的 A 路线也会用源动作的重力特征和稳定高处足点推断平台支撑，判定成立即参与精修和终验。 [场景支撑排查](docs/VALIDATION.md#按剩余问题处理)

离线自碰与平台碰撞配对排除脚底胶囊。加载配套资产进行物理仿真时，每脚 7 个胶囊参与接触计算。

## 精修路线和预算

默认精修路线为 A。FRAX 初解按源动作的支撑类型分流，分别保留接触式与可信非脚支撑的 Fix 配置。中段精修按接触相位与残差派发常规专项、窗口联解或回溯。 [转换流程](docs/PIPELINE.md)

- `refine` 的默认预算为 `0.7 秒 × 50 Hz 输入帧数`，只用于中段专项精修。FRAX 初解、末尾时序（含去抖）、限速补帧和独立终验分别计时。批次外层默认无总时限。
- 加 `--repair-budget unlimited` 取消中间专项精修的总时限。两种模式都保留单次候选时限。到达上限、已通过或可执行候选用尽时，按已复检的轨迹收尾并完成终验。
- 需要限制整次转换或整个批次时，可用 `--budget` 指定总秒数，覆盖初解、精修和终验。 [批处理与预算](docs/BATCH.md)

### 精修预算用完后续跑

报告显示耗尽专项精修预算且仍有待办时，沿用源文件、模型和转换选项。用 `--continue-from` 指向报告中的检查点，输出到新目录。省略预算参数给新一轮默认额度，加 `--repair-budget unlimited` 可继续搜索到可执行候选用尽。

```sh
dex-frax convert /path/to/input.bvh outputs/bvh_round2 \
  --source mocap --mode refine --bvh-units cm --bvh-translations absolute \
  --continue-from outputs/bvh_refine/frax --repair-budget unlimited
```

如果报告将检查点标为 `pre_final_retime`，使用 `outputs/bvh_refine/frax/pre_final_retime`。 [读报告与处理残余问题](docs/VALIDATION.md)

## 使用文档

- [当前版本和适用范围](docs/CURRENT_VERSION.md)
- [输入、接触与输出字段](docs/INPUTS.md)
- [处理流程](docs/PIPELINE.md) · [编号流程图](docs/REFINE_FLOWCHART.html)
- [候选采用与验收规则](docs/SAFETY_REPAIR_RULES.md)
- [批处理与续跑](docs/BATCH.md)
- [第三方来源与许可](THIRD_PARTY.md)
