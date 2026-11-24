# 输入、接触与输出

## BVH

`--source` 可选 `lafan`、`soma`、`mocap` 或 `auto`。单位用 `--bvh-units cm|m|mm`，位置通道约定用 `--bvh-translations absolute|offset`。默认 `cm/absolute`。使用其他约定时显式指定。默认脚接触 v2 可改用 v1 做同输入对照。

使用 LaFAN 或 SOMA 时，准备相应的 `lafan1_surfaces.json` 或 `soma_surfaces.json`，将 `DEX_FRAX_SOURCE_GEOMETRY` 指向文件所在目录。发行包中的对应输入要依赖这个目录。 [几何来源与配置](../src/dex_frax/v8/reference/SOURCE_GEOMETRY.md)

BVH 和 SMPL-X 在重采样前检查整条原生序列，在输出采样网格上再查一遍。邻域跳变使用 4 倍步幅及 0.65 m/s、120°/s 的工程去噪底线。50 Hz 下这两个底线相当于每帧约 1.3 cm 和 2.4°。原生跳变与处置写在 `motion.source_review.json` 的 `events`。50 Hz 复检将邻域证据和对应的已检查原生帧对照。确认由重采样引入的跳变时，固定两端并以原生帧重建。无法重建的异常会阻止训练导出。处置记录保存在 `resampled`，包含采样帧号和对应原帧号。

- **首尾异常。** 与内侧连续动作明显断开的端点帧从工作序列剔除。
- **中间异常。** 检查单帧与连续多帧异常，也检查发生在单个关节或多个关节的跳变。用两侧姿态作固定端点，位移线性插值，转角用 SLERP。已满足成对往返条件的单帧尖峰用紧邻的两帧作锚点，按成对判据试修并复检接缝。短段试修范围随原生帧率换算，按约 0.2 秒计算且至少允许 3 帧，例如 30 Hz 为 6 帧，50 Hz 为 10 帧。整帧或单关节重复停滞时，用恢复后的位移与两侧速度交叉验证，符合证据才重建。固定骨架通道按其约束处理。
- **待复核。** 连续的快速动作、证据不足的轻微提示和未通过复检的候选保留原始样本。`motion.source_review.json` 记录各处采用或保留的理由，具有相邻样本佐证且未解决的跳变会阻止训练导出。仅达到宽邻域候选门槛的变化保留供复核。同向快慢变化保留原始姿态，达到幅度筛查门槛时记为复核提示。该策略要求段内各步与前后邻步的向量内积均为正，且前后邻步也同向。人体速度依据和固定骨架、采样丢失等证据分别检查。

相邻或重叠的异常在 `motion.source_review.json` 的 `anomaly_segments` 中按原始帧归并，列出涉及的通道、修复帧和待复核项。工作序列的裁剪与修复均保留原始文件和源帧号。轨迹的 `source_frame_float` 可追溯到原帧；仅截取短输出时，源审查仍检查整条输入。

### 人体速度复核

原始与重采样的人体速度单独记录在各自的 `human_speed`。超出参考线只生成复核记录。报告列出实际使用的测点及条件不足的项目。

| 测量 | 复核值 | 适用范围 |
| --- | --- | --- |
| 根部水平速度 | 12.34 m/s | 声明整条为无助力地面跑动时应用。单条用 `--ground-running`，BVH 批次可在对应清单条目声明 `"ground_running": true`。 [世界田联表 D](https://worldathletics.org/download/downloadnsa?filename=31514203-6724-455e-906c-ef931384c656.pdf&urlslug=biomechanical-analysis-of-the-sprint-and-hurd) |
| 骨盆竖轴转速 | 2,052°/s | 只比较直立且与躯干同步的竖轴旋转。数值由整人冰上自转 342 rpm 换算。 [吉尼斯纪录](https://www.guinnessworldrecords.com/world-records/fastest-spin-ice-skating) |
| 伸膝角速度 | 2,200°/s | 用髋、膝、踝测点及大腿姿态估计屈伸轴，只比较伸展方向。原始研究最高散点约 2,170–2,180°/s。 [踢球研究图 6](https://link.springer.com/article/10.1007/s12283-024-00458-4#Fig6) |
| 手脚末端世界速度 | 25 m/s | 工程复核参考。测点由模型提供，例如 SMPL-X 手腕和前脚掌。 [拳击研究图 1](https://ojs.ub.uni-konstanz.de/cpa/article/download/4491/4179) · [足球脚趾速度表 2](https://www.jssm.org/volume01/iss3/cap/jssm-01-72.pdf) |

机器人输出的根部孤立跳变、额外下坠和精修速度余量分别配置，当前各自使用 3.5 cm 的等效初值。精修速度余量按采样率换算，30 Hz 为 1.05 m/s，50 Hz 为 1.75 m/s。29 轴速度按目标模型 URDF 终验。

## SMPL-X

用 `--source smplx`。SMPL-X 人体几何由独立解释器准备，之后交给 Robot-Retarget 求解。本机可复用已有的 SMPL-X 环境和模型权重：

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
| `motion.npz` | 50 Hz 根部位置、姿态和 29 轴轨迹，优先用于回放和交换 |
| `motion.pkl` | 同一条轨迹的 PKL 格式，供本地工具读取。只加载自己生成或可信来源的文件。 |
| `source_time_s`、`source_frame_float` | `motion.npz` 中的源时间和原生帧号。存在独立时间适配记录时另存 `motion.time_map.npz`。 |
| `motion.source_review.json` | 原生输入与重采样的审查、修复及待复核位置 |
| `motion.stage_*.npz`、`motion.full_chain.json` | 初解、落地及触发时的保护式 Frax 各阶段 |
| `repair.json` | 局部候选的执行顺序和采纳结果。B、C 的 V8 段另见 `cascade_segments.json`。 |
| `pipeline.md`、`quality.json`、`run.json` | 可读处理报告、独立终验和运行记录 |
| `training.npz` | `refine` 质量标记为空时生成的训练轨迹 |
