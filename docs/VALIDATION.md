# 查看结果与排查

转换完成后，先读输出目录下的 `frax/pipeline.md` 查看质量状态、精修路线和未解决事项。`frax/quality.json` 提供验收标记和各项测量数据。也可从已有记录重新显示报告。

```sh
dex-frax explain outputs/bvh_refine/frax
```

## 按验收状态处理

| `quality.json` 状态 | 文件和用途 | 接下来的操作 |
|---|---|---|
| `accepted` | `refine` 导出 `motion.npz` 和 `training.npz`。 | 用前者回放，用后者接入训练数据流程。 |
| `accepted_with_notes` | 保留可回放轨迹，终验仍有姿态相似度备注。 | 查看 `pose_notes`，对照源姿态及相关帧。 |
| `review` | 预览轨迹，可能残留自碰撞、穿地、支撑或时序问题。 | 使用 `motion.npz` 回放，按 `flags` 对照问题帧。 |

`run.json` 的 `status=complete` 表示转换进程已完成。能否用于训练以验收状态和实际导出的 `training_npz` 为准。普通轨迹和训练文件的字段见 [输出说明](INPUTS.md#输出)。

`head_tracking.target_clipped_frames_by_joint` 记录各头部关节源目标截限的帧数，`orientation_error_deg` 记录头部朝向与源目标的差异。触限饱和不影响验收状态。实际输出的角度、超速和碰撞问题按 `flags` 排查。 [头部硬限位规则](SAFETY_REPAIR_RULES.md#头部硬限位)

`dex-frax preview` 用 `motion.pkl` 渲染 MP4 和同名 GIF。安装时加上 `preview` 可选依赖，再执行下列命令。预览和续跑会读取 PKL，只使用自己生成或可信来源的文件。

```sh
dex-frax preview outputs/bvh_refine/frax/motion.pkl --output outputs/preview.mp4
```

## 按剩余问题处理

| 情况 | 要看什么 | 操作 |
|---|---|---|
| 精修总时限到期，仍有待办 | `pipeline.md` 的「精修状态」与 `repair.json` 的 `shared_budget_exhausted` | 用同一构建、输入、模型和原选项，从报告中的检查点追加 `--repair-budget` 并写入新目录。 |
| 单次候选超时 | `repair.json` 的窗口和原因。 | 仍有超时任务时可续跑重试。单次求解上限仍适用。 |
| 候选拒收或候选已试尽 | `repair.json` 的候选原因及 `quality.json` 的残余问题。 | 根据对应帧检查穿地、碰撞、支撑或时序。是否续跑取决于还有没有可尝试的候选。 |
| 源数据异常或场景支撑待确认 | `motion.source_review.json`、`quality.json`，以及 `motion.json` 中的接触和场景信息。 | 核对源动作、帧率、单位、骨架和接触。需要改变输入或场景参数时，从原始输入转换到新目录。 |

续跑沿用原转换选项。采用最终限速补帧时，报告会给出 `pre_final_retime` 检查点。 [续跑命令](../README.md#精修预算用完后续跑) · [批次恢复](BATCH.md#续跑与诊断)

## 定位问题帧

`motion.npz` 中的 `source_frame_float` 和 `source_time_s` 对应原生数据。机器人输出帧按 50 Hz 计，源帧按输入实际帧率计，两个序列都从 0 起算。检查同一处异常时，使用报告的源帧映射对齐原始动作和机器人轨迹。 [输入、时间映射与输出](INPUTS.md)

这套终验用于离线运动学和几何。接入真实硬件前，需要在实际控制环境验证动作可执行性。 [验收规则](SAFETY_REPAIR_RULES.md)
