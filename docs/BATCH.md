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

`dex-frax-pool` 支持 Fast 和 Refine A，省略 `--mode` 时使用 Refine A。每个 worker 启动时同时分配一个 CPU 槽和 2.75 GiB 基础内存。Linux 优先使用独立物理核，物理核占满后可使用其余逻辑核，并参考外部 CPU 负载调整准入。加载、FRAX 分块和复检按阶段申请增量，临时额度在安全阶段边界归还。Refine A 的专项精修也使用共享池，单条精修使用默认额度或 `--repair-budget unlimited`。

```sh
dex-frax-pool manifest.json outputs/pool-fast --mode fast
dex-frax-pool manifest.json outputs/pool-fast --mode fast --resume
dex-frax-pool manifest.json outputs/pool-refine --mode refine
```

共享池按顺序处理在跑 worker 的必要内存请求、待启动动作、算子的可选并行请求。编译仍逐个派发。已有 worker 仅等待编译时，只要 CPU 和基础内存额度充足，就可以继续启动新任务。

Fast 的初始可用预算为启动时主机可用内存减 2 GiB。运行中扣除已发放但尚未使用的额度，继续检查 2 GiB 整机余量。Refine 默认低于 8 GiB 暂停新增额度，持续回到 10 GiB 后恢复，启动新 worker 或发放可选并行资源后至少剩 12 GiB。两种模式启动新 worker 时，Linux cgroup 至少剩 1 GiB。可用 `--memory-watermarks DANGER RECOVER START` 指定三个递增的整机水位，单位为 GiB。

Fast 的 2.75 GiB 基础额度用于冷启动。后续安全阶段按实际驻留内存、阶段工作区和动作帧数申请和归还额度。冷编译、首次求解块和外部 SMPL-X 预处理保留额外保障。每个运行项的 `grant_bytes`、`resident_bytes`、`unused_bytes` 分别记录当前额度、实占和未用额度。

Fast 优先用空闲 worker 接下一条动作，复用模型和 FRAX 编译缓存，按热加载的增量需求领取额度。暂时开不出新进程时，空闲 worker 可以等候接单。队列清空、出现内存压力或在途任务需要额外内存时及时回收，持续空闲 60 秒也会回收。

待发动作多于在途动作时，按 CPU 和内存余量增加并发。待发动作已可由在途进程再各接一条时，优先等热进程。等待窗口取同类输入最近至多五次冷启动到首个 FRAX 求解块的中位用时，最长 60 秒。窗口内无任务接单，就允许新开一路。热进程可以随时接单。

A 的全片几何复检默认串行。加 `--audit-workers 2` 可启用并行实验。队列为空且没有必要内存请求等待时，96 帧及以上的全片复检可申请一个额外 CPU 槽和辅助进程内存。辅助进程用自己的 MuJoCo 状态检查全身自碰，主进程同时检查手臂碰撞和原帧及帧间穿地。两边汇总后沿用同一终验。小范围增量复检沿用本地缓存。资源不足时按串行复检，辅助进程退出后归还额度。Linux 将辅助进程绑定到获准的 CPU，macOS 由系统安排核位。

第一个通过资源准入的 worker 准备基础模型和带脚部凸包的全身碰撞模型，后续 worker 共享经校验的缓存路径。二进制缓存按资产、构建代码和运行库版本定位，建立时核对 XML 编译模型与二进制模型的数组，加载时核对哈希。每个进程各自加载模型，每次验收创建独立的运行状态并重算轨迹采样。辅助进程按模型文件大小预留读入缓冲与解码后的模型内存，并计入进程基础消耗与逐帧工作区。

`status.json` 记录进度、CPU 分配和内存等待，`memory_events.jsonl` 记录发放、回收、停发和恢复。`model_cache.json` 记录缓存准备时间及命中情况。每个 `attempt_*` 下的 `geometry_timings.jsonl` 记录全片几何复检的主进程耗时，以及辅助进程的启动、输入接收、模型准备、扫描、结果回传、退出和资源归还耗时。并行两路的时间有重叠。

每条动作的结果位于 `<id>/attempt_*/frax`。续跑沿用原清单、模式和转换选项，使用输入、目标机器人资产和产物哈希验证已完成的输出。Fast 输出预览用的 `motion.npz`，中断项重新转换。Refine A 的未完成项可从最近的有效检查点继续，沿用本轮已用额度，也可在清单里为单条动作填 `continue_from` 指向同一输入的可信检查点。共享池清单还支持 `source: "smplx"`，输入适配参数与 `dex-frax convert` 相同，例如 `--input-fps 30 --grounding contact`。

## 续跑与诊断

常规批次使用原清单、原输出目录和原转换选项追加 `--resume`。核对输入、源码和模型后，已完成项可以保留。单条动作可用 `convert --continue-from` 从报告指定的检查点接续并写入新目录。`--recheck-from` 在兼容的代码更新后重新验收和生成候选队列。仍有残差的 V8 超时段可在续跑时重试。

批次状态为完成时，逐条查看 `frax/pipeline.md` 的「精修状态」。若报告提示专项精修预算耗尽且仍有任务，按原输入和转换选项用 `--continue-from` 接续该动作。省略预算参数开启一轮默认额度，显式加 `--repair-budget unlimited` 开启无限额度。若报告只记录了单窗超时或候选拒收，查看 `repair.json` 中对应的窗口和原因。 [单条续跑示例](../README.md#精修预算用完后续跑)
