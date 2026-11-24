# Dex 29 关节机器人资产

本目录随 Git 分发，包含匹配的 `model.json`、`mjcf/dex_29.xml`、`meshes/`、`meshes_visual/`、`meshes_convex/`、`urdf/` 及原始 `LICENSE`。使用前保留原始许可；资产使用 OpenAtom Open Hardware License 1.0，不能用本仓库代码的 MIT 许可替代。

机器人原始资产由 UBTECH 发布。本目录随附的模型包来自 Dex-Video2Robot 的固定快照，包含其整理后的 29 轴 URDF、生成的 MuJoCo 模型和视觉及凸包网格。Dex-RetargetPro 沿用该快照中的模型、网格及许可证。 [UBTECH 原始资产](https://github.com/UBTECH-Robot/Walker_TienKung_DEX_URDF/tree/d172cfb8cdb4961abeda38bb363d555e59db4531) · [Dex-Video2Robot 资产快照](https://github.com/luckylhf/Dex-Video2Robot/tree/cf252a2172344908dfa5ef5901e3d78506bbea6a/Dex-GMR/assets/dex)

- 模型 XML SHA-256：`17573c55d1c9d9d43b2e44a5e3ebc237027bfcfb6f5b5d928aa0205792548194`。
- `model.json` 的 `model_sha256` / `urdf_sha256` 用于转换、训练、回放和缓存身份校验；模型不匹配时应重新转换、训练和验证，不要只改 JSON 哈希绕过检查。
- 转换和回放默认使用本目录。若已有完整且经过核验的资产副本，可设置 `DEX_FRAX_ASSETS=/absolute/path/to/assets/dex_evt` 指向该副本。该目录须包含上述模型、网格和许可文件。
