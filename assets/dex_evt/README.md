# Dex 31 轴机器人资产

本目录从 `Tienkung_Dex_Model_All` 导入基础版。原始 URDF、MJCF、网格和 [OpenAtom 许可证](LICENSE) 保留上游内容与相对路径。[model.json](model.json) 记录导入提交、模型哈希、关节限位和导出顺序。

| 用途 | 上游文件或目录 | 转换器映射 |
| --- | --- | --- |
| 运动学和额定速度 | `urdf/tiangong3.urdf`、`meshes/` | Frax 直接读取 URDF，按原始关节名排列 |
| 模型、碰撞网格和执行器 | `mujoco/tiangong3_torq.xml`、`mujoco/meshes_convex/` | MuJoCo 读取原始模型。导出世界的地面平移至 `z=0`，骨盆默认位姿同步平移 |
| 视觉网格和离线表面 | `mujoco/meshes_simplify/` | 按同名 `*_visual` 几何映射到减面网格，保留转换器使用的表面。模型内的惯性取映射前的上游值 |
| 脚底碰撞 | 两侧 `ankle_roll_*_link` 各 7 个 `capsule` | 仿真使用胶囊。离线落脚、自碰和场景检测使用脚部视觉网格凸包 |
| 导出关节顺序 | 原始 31 个活动关节 | 保留原身体 29 列顺序，末尾为 `head_yaw_joint`、`head_pitch_joint` |

求解器的身体角色映射和具名装配排除在 [robot_profile.py](../../src/dex_frax/robot_profile.py) 中。离线几何映射在 [model_cache.py](../../src/dex_frax/model_cache.py) 中。可用 `DEX_FRAX_ASSETS` 指向按同一布局导入的资产目录。

从已检出的上游工作树重新导入：

```sh
python assets/dex_evt/tools/build_assets.py --source /path/to/Tienkung_Dex_Model_All
```
