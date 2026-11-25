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

`--budget` 是批次总时限。`--repair-budget` 覆盖每条动作的默认精修时限，也就是整条输入按 50 Hz 换算的总帧数 × 0.5 秒。各精修阶段共用该时限，单次专项求解最多 20 秒。精修时限耗尽时保留已采用结果和待处理记录，随后独立终验。并发受内存和磁盘准入限制，磁盘保留量由 `--disk-reserve-gib` 配置，默认 80 GiB。

批处理也可指定 `--repair-budget unlimited`，取消每条动作的精修总时限。批次总时限 `--budget`（默认 1,800 秒）和显式指定的 `--frax-timeout` 仍会限制实际运行时间。

默认精修路线为 A，按专项轮转。 [转换流程](PIPELINE.md)

`--segmented` 把已校验的 Frax 初解落盘后交给单独的精修进程。`--capacity-scheduler` 可按 CPU 和内存预算派发初解和精修任务。用 `dex-frax batch --help` 查看资源配置参数。

## 续跑与诊断

常规批次使用原清单、原输出目录和原转换选项追加 `--resume`。核对输入、源码和模型后，已完成项可以保留。单条动作可用 `convert --continue-from` 从报告指定的检查点接续并写入新目录。`--recheck-from` 在兼容的代码更新后重新验收和生成候选队列。仍有残差的 V8 超时段可在续跑时重试。

批次状态为完成时，逐条查看 `frax/pipeline.md` 的「精修状态」。若报告提示该条动作的精修总预算耗尽且仍有任务，按原输入和转换选项用 `--continue-from` 接续该动作，并用 `--repair-budget` 指定下一轮新增的秒数。若报告只记录了单窗超时或候选拒收，先查看 `repair.json` 中对应的窗口和原因。 [单条续跑示例](../README.md#精修预算用完后续跑)
