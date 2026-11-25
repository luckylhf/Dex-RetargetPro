# 第三方来源与许可

代码、机器人资产和人体模型分别遵循各自许可。

## Frax

源码包和 wheel 安装时从 Python 包索引获取 Stanford ASL 的 `frax[ik]==0.2.0`。 [Frax 依赖](https://pypi.org/project/frax/0.2.0/)

MIT 许可及版权声明见 [FRAX_LICENSE](docs/FRAX_LICENSE)。每次转换记录实际导入的 Frax Python 源码哈希。

## 转换与检查组件

`src/dex_frax/v8` 及部分姿态、速度检查模块源自 Dex-SuperMimic 的 V8 转换组件。继承版本为 `V8_cpu_coupled_arms_20261105`，来源记录见 [provenance.json](src/dex_frax/v8/provenance.json)，继承许可保留于 [LICENSE](LICENSE)。本项目对输入适配、求解和修复作了修改，当前代码与原始快照不同。

GMR 和 NVIDIA 的声明保留于 `src/dex_frax/v8/reference/`。

SOMA 校准使用 NVIDIA SOMA Retargeter 提交 `cbdb2b494a2af82137bf9b6f15122d60167281c4` 的两个文件。`soma_zero.bvh` 是上游单帧校准文件 `soma_zero_frame0.bvh` 的副本。`soma_contact_config.json` 提供脚部标记和接触参数。 [零姿态校准](https://github.com/NVIDIA/soma-retargeter/blob/cbdb2b494a2af82137bf9b6f15122d60167281c4/soma_retargeter/assets/soma/soma_zero_frame0.bvh) · [接触配置](https://github.com/NVIDIA/soma-retargeter/blob/cbdb2b494a2af82137bf9b6f15122d60167281c4/soma_retargeter/assets/soma/contact_processing/soma_contact_config.json) · [NVIDIA 许可证](src/dex_frax/v8/reference/NVIDIA/LICENSE)

## 机器人资产

`assets/dex_evt/` 包含独立的 `LICENSE`、来源清单、URDF、MJCF 和网格。源码分发包保留完整目录及其 OpenAtom Open Hardware License 1.0。使用 Python wheel 时，通过 `DEX_FRAX_ASSETS` 指向另备的匹配资产目录。 [机器人资产说明](assets/dex_evt/README.md)

## 人体表面几何与模型

`lafan1_surfaces.json` 和 `soma_surfaces.json` 是外置人体表面几何表。使用者按对应来源条款准备所需的表，并把 `DEX_FRAX_SOURCE_GEOMETRY` 指向表所在目录。 [来源及适用条款](src/dex_frax/v8/reference/SOURCE_GEOMETRY.md)

SMPL-X 人体模型参数不随项目分发，须从 [SMPL-X 官方网站](https://smpl-x.is.tue.mpg.de/) 按其许可获取。

## 输入数据

人体动作及其转换产物遵循对应输入数据的使用和分发条款。 [LAFAN1 许可](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt)
