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

## 头部动作

头部 yaw、pitch 跟踪人体头部相对躯干的朝向，导出关节名末两列为 `head_yaw_joint` 和 `head_pitch_joint`。源目标超出可达角度时，在机器人限位内尽量跟踪，触限饱和不报错。BVH 缺失头部关节时，头部目标取躯干相对零位。跟踪差异和校准来源见 `frax/quality.json` 的 `head_tracking`。 [头部硬限位规则](docs/SAFETY_REPAIR_RULES.md#头部硬限位)

## 脚部几何与碰撞检查

离线转换用脚部视觉网格的整脚凸包确定落脚高度，检查脚地穿透、脚与脚或腿的自碰，以及 `--scene` 中的脚与平台碰撞。

离线自碰与平台碰撞配对排除脚底胶囊。加载配套资产进行物理仿真时，每脚 7 个胶囊参与接触计算。

## 精修路线和预算

默认精修路线为 A，按专项轮转处理腿部避碰、脚部支撑、穿地、局部抖动、身体支撑和手臂避碰。 [转换流程](docs/PIPELINE.md)

默认精修时限为原始整条动作换算到 50 Hz 的总帧数 × 0.5 秒。`--repair-budget` 可指定秒数，`--repair-budget unlimited` 可取消共享时限。单次求解上限仍然适用。`--budget` 限制整次转换或整个批次，默认分别为 24 小时和 1,800 秒。 [批处理与预算](docs/BATCH.md)

### 精修预算用完后续跑

当 `frax/pipeline.md` 的「精修状态」显示共享时限耗尽且还有任务，在同一构建下沿用源文件、模型和转换选项。用 `--continue-from` 指向报告中的检查点，用 `--repair-budget` 指定本轮时限。

```sh
dex-frax convert /path/to/input.bvh outputs/bvh_round2 \
  --source mocap --mode refine --bvh-units cm --bvh-translations absolute \
  --continue-from outputs/bvh_refine/frax --repair-budget 900
```

如果报告将检查点标为 `pre_final_retime`，使用 `outputs/bvh_refine/frax/pre_final_retime`。 [读报告与处理残余问题](docs/VALIDATION.md)

## 使用文档

- [当前版本和适用范围](docs/CURRENT_VERSION.md)
- [输入、接触与输出字段](docs/INPUTS.md)
- [处理流程](docs/PIPELINE.md) · [编号流程图](docs/REFINE_FLOWCHART.html)
- [候选采用与验收规则](docs/SAFETY_REPAIR_RULES.md)
- [批处理与续跑](docs/BATCH.md)
- [第三方来源与许可](THIRD_PARTY.md)
