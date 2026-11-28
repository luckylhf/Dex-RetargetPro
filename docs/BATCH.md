# 批处理与续跑

## 清单

内置 `batch` 接受 BVH 输入，`source` 可填 `auto`、`lafan`、`soma` 或 `mocap`。SMPL-X 输入使用 `convert --source smplx`。批处理使用 JSON。每条记录包含唯一 `id`、`input`、`source` 和源文件的 SHA-256。路径可相对清单。以下命令用一条 mocap BVH 生成清单。

```sh
python - /path/to/input.bvh <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
manifest = {'cases': [{'id': 'motion_01', 'input': str(source),
                       'source': 'mocap', 'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}]}
Path('manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
```

## 模式和预算

```sh
dex-frax batch manifest.json outputs/batch-refine \
  --mode refine --workers auto
```

`refine` 批次默认没有整批总时限，每条动作给中间专项精修 `0.7 秒 × 50 Hz 输入帧数`。可用 `--repair-budget unlimited` 取消专项总时限，也可用 `--budget` 限制整批运行时间、用 `--frax-timeout` 限制单条 worker 时间。专项时限耗尽时保留已采用结果和待处理记录，随后完成收尾和独立终验。

常规专项候选单次限时 20 秒，A 路线的支撑相位腿部联解、支撑相位回溯和推断平台候选单次限时 40 秒。有可继续的进展时，时间片结束后可重新派发。候选处理完毕或没有可执行候选时进入收尾和终验。并发受内存和磁盘准入限制，`--disk-reserve-gib` 的默认值为 80 GiB。

专项时钟在初解和精修前检查结束后启动，搜索停止时结束。末尾去抖、限速补帧与独立终验分别计时。`repair.json` 记录预算范围、等效输入帧数和专项耗时。

默认精修路线为 A，按接触相位和残差派发候选。 [转换流程](PIPELINE.md)

`--segmented` 把已校验的 Frax 初解落盘后交给单独的精修进程。`--capacity-scheduler` 可按 CPU 和内存预算派发初解和精修任务。用 `dex-frax batch --help` 查看资源配置参数。

## CPU 与内存共享池

`dex-frax-pool` 使用 Refine A 处理带哈希的清单。每个 worker 启动时同时分配一个 CPU 槽和 2.75 GiB 基础内存。Linux 优先使用独立物理核，并参考当前的外部 CPU 负载调整准入。加载、初解分块、专项精修和复检在分配前申请增量，临时额度在安全阶段边界归还。单条专项精修使用相同的默认额度或 `--repair-budget unlimited`。从本轮检查点自动恢复时沿用已用额度。

```sh
dex-frax-pool manifest.json outputs/pool-refine --workers auto
dex-frax-pool manifest.json outputs/pool-refine --workers auto --resume
```

共享池按顺序处理在跑 worker 的必要内存请求、待启动动作、算子的可选并行请求。扣除已发放但尚未使用的额度后，整机余量默认低于 8 GiB 暂停新增额度，持续回到 10 GiB 后恢复。启动新 worker 或发放可选并行资源后，整机默认至少剩 12 GiB，Linux cgroup 至少剩 1 GiB。可以用 `--memory-watermarks DANGER RECOVER START` 为受控运行指定整机水位，单位为 GiB。

A 的全片几何复检默认串行。加 `--audit-workers 2` 可启用并行实验。队列为空且没有必要内存请求等待时，96 帧及以上的全片复检可申请一个额外 CPU 槽和辅助进程内存。辅助进程用自己的 MuJoCo 状态检查全身自碰，主进程同时检查手臂碰撞和原帧及帧间穿地。两边汇总后沿用同一终验。小范围增量复检沿用本地缓存。资源不足时按串行复检，辅助进程退出后归还额度。Linux 将辅助进程绑定到获准的 CPU，macOS 由系统安排核位。

第一个通过资源准入的 worker 准备基础模型和带脚部凸包的全身碰撞模型，后续 worker 共享经校验的缓存路径。二进制缓存按资产、构建代码和运行库版本定位，建立时核对 XML 编译模型与二进制模型的数组，加载时核对哈希。每个进程各自加载模型，每次验收创建独立的运行状态并重算轨迹采样。辅助进程按模型文件大小预留读入缓冲与解码后的模型内存，并计入进程基础消耗与逐帧工作区。

`status.json` 记录进度、CPU 分配和内存等待，`memory_events.jsonl` 记录发放、回收、停发和恢复。`model_cache.json` 记录缓存准备时间及命中情况。每个 `attempt_*` 下的 `geometry_timings.jsonl` 记录全片几何复检的主进程耗时，以及辅助进程的启动、输入接收、模型准备、扫描、结果回传、退出和资源归还耗时。并行两路的时间有重叠。

每条动作的结果位于 `<id>/attempt_*/frax`。续跑沿用原清单与转换选项，使用输入、目标机器人资产和产物哈希验证已完成的输出，未完成项从最近的有效检查点续跑。也可在清单里为单条动作填 `continue_from`，指向同一输入的可信检查点。共享池清单还支持 `source: "smplx"`，输入适配参数与 `dex-frax convert` 相同，例如 `--input-fps 30 --grounding contact`。

## 续跑与诊断

常规批次使用原清单、原输出目录和原转换选项追加 `--resume`。核对输入、源码和模型后，已完成项可以保留。单条动作可用 `convert --continue-from` 从报告指定的检查点接续并写入新目录。`--recheck-from` 在兼容的代码更新后重新验收和生成候选队列。仍有残差的 V8 超时段可在续跑时重试。

批次状态为完成时，逐条查看 `frax/pipeline.md` 的「精修状态」。若报告提示专项精修预算耗尽且仍有任务，按原输入和转换选项用 `--continue-from` 接续该动作。省略预算参数开启一轮默认额度，显式加 `--repair-budget unlimited` 开启无限额度。若报告只记录了单窗超时或候选拒收，查看 `repair.json` 中对应的窗口和原因。 [单条续跑示例](../README.md#精修预算用完后续跑)
