# 3_Data/ - 模拟数据目录

**目的**: 存放模拟过程中产生的数据或用于分析的外部公开数据。

### 数据文件说明

| 文件名 | 描述 | 大小 |
| :--- | :--- | :--- |
| `simulation_2body_leapfrog.csv` | 两体系统（太阳-地球）Leapfrog积分模拟数据 | ~31MB |
| `simulation_2body_euler.csv` | 两体系统（太阳-地球）Euler积分模拟数据 | ~31MB |
| `convergence_test.csv` | 不同步长下的收敛性测试结果 | <1KB |

### 未上传的数据文件

以下文件因超过GitHub的50MB文件大小限制而未上传：

| 文件名 | 描述 | 大小 |
| :--- | :--- | :--- |
| `simulation_3body_leapfrog.csv` | 三体系统（太阳-木星-小行星）Leapfrog积分模拟数据 | ~230MB |

### 如何重新生成数据

1. 确保已安装依赖：`pip install -r ../2_Code/requirements.txt`
2. 进入代码目录：`cd ../2_Code`
3. 运行主程序：`python main.py`

程序将自动生成所有数据文件和图表。

### 数据格式

CSV文件格式：
- 第一行：表头（time, body1_x, body1_y, body1_z, body1_vx, body1_vy, body1_vz, ...）
- 后续行：每一行对应一个时间步的所有天体位置和速度

单位系统：
- 长度：天文单位（AU）
- 质量：太阳质量（M☉）
- 时间：年（yr）