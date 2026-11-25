# 输入、接触与输出

## BVH

`--source` 可选 `lafan`、`soma`、`mocap` 或 `auto`。用 `--bvh-units cm|m|mm` 指定单位，用 `--bvh-translations absolute|offset` 指定位置通道是绝对值还是相对 `OFFSET` 的增量。默认使用厘米和绝对位置。

使用 LaFAN 或 SOMA 时，准备相应的 `lafan1_surfaces.json` 或 `soma_surfaces.json`，将 `DEX_FRAX_SOURCE_GEOMETRY` 指向文件所在目录。 [人体表面几何配置](../src/dex_frax/v8/reference/SOURCE_GEOMETRY.md)

## 输入审查

BVH 和 SMPL-X 均检查整条原生序列及重采样结果。限定 `--max-frames` 时，原生审查仍覆盖完整输入。转换可剔除与连续动作断开的首尾帧，或在通过接缝复检后修复有充分依据的中间异常。影响训练的未解决跳变会阻止训练导出。

`motion.source_review.json` 记录原生异常、重采样处置及待复核位置。人体速度提示见 `human_speed`。仅在输入确属无助力地面跑动时，用 `--ground-running` 启用水平跑速复核。BVH 批次可在清单条目中设置 `"ground_running": true`。机器人 31 轴的输出速度按目标 URDF 独立终验。

## SMPL-X

用 `--source smplx`。人体几何由选定的 Python 环境准备，再交给 Dex-RetargetPro 求解。使用源码目录中的 `./dex-frax` 启动器时，可复用本机环境和模型权重。

- `.venv/smplx-env` 可软链接到含 PyTorch 和 SMPL-X 的 Python 环境。启动器默认使用其中的 `bin/python`。
- `.venv/smplx-models` 可软链接到模型父目录。模型文件放在其 `smplx/` 下，例如 `smplx/SMPLX_NEUTRAL.npz`。

也可通过 `SMPLX_PYTHON`、`SMPLX_MODEL_DIR` 设置默认值，或在命令中显式传入 `--smplx-python`、`--body-models`。输入缺少坐标系或帧率时，用 `--up-axis` 或 `--input-fps` 补齐。

```sh
dex-frax convert /path/to/input.pt outputs/smplx_preview \
  --source smplx --mode fast --input-fps 30 --grounding contact \
  --body-models /path/to/body_models
```

## 机器人模型与接触

源码包的模型在 `assets/dex_evt`，包含可视 STL、碰撞几何及模型配置。安装后用 `DEX_FRAX_ASSETS` 指向匹配资产目录。`refine` 可用 `--scene` 提供机器人导出坐标系下的实测轴对齐箱体。存在 GVHMR 静止分数时，`--grounding contact` 先用高置信、位置较低的脚估计共同的高度漂移。最终足部接触仍由脚跟和脚尖相对地面的高度、速度、稳定帧数和脚锚位移推断，输出全脚掌、脚尖、脚跟或摆动。可信身体支撑根据固定、滚动或滑动相位分别约束。腾空和未确认的抬高支撑进入高度保护。

## 输出

| 文件 | 用途 |
|---|---|
| `motion.npz` | 50 Hz 根部位置、`xyzw` 姿态和 31 列 `dof_pos`，用于交换和回放 |
| `motion.pkl` | 同一条轨迹的 PKL 格式，供本地工具读取。只加载自己生成或可信来源的文件。 |
| `source_time_s`、`source_frame_float` | `motion.npz` 中的源时间和原生帧号。存在独立时间适配记录时另存 `motion.time_map.npz`。 |
| `motion.source_review.json` | 原生输入与重采样的审查、修复及待复核位置 |
| `motion.stage_*.npz`、`motion.full_chain.json` | 初解、落地及触发时的保护式 Frax 各阶段 |
| `repair.json` | 局部候选的执行顺序、采纳结果和待处理项 |
| `pipeline.md`、`quality.json`、`run.json` | 可读处理报告、独立终验和运行记录 |
| `training.npz` | `refine` 通过终验后导出的训练轨迹，包含 31 列 `joint_pos` 和 `wxyz` 根部四元数 |

两个 NPZ 均用 `joint_names` 标明列顺序。最后两列依次为 `head_yaw_joint`、`head_pitch_joint`。
