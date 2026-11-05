# 与旧 Halcon C++ 项目一致的颈部弯曲角

本页记录此前的兼容实现。当前默认已改为[几何方向范围](robust-neck-angle.md)，
以下旧口径通过 `measure_cpp_bending(contour)` 或 `measure_direction_angles(angles)` 显式调用：

1. 按中心线顺序读取法向角，转换为 C++ `float` 对应的 float32。
2. 每个窗口顺序累加 10 个角度，再除以 10。
3. 按原循环 `k < N-10` 生成 **N−10 个窗口**，不补上最后一个窗口。
4. 计算平滑角的最大值减最小值，转为度，再保存为 float32。

这对应旧项目 `OneSperm.cpp` 第 887–903 行。11 点以下没有可计算结果；
恰好 11 点只有一个窗口，因此范围为 0。保持旧代码的原始周期角，不做 unwrap，
也不将结果强制折回 0–90° 或 0–180°。跨越角度周期边界时可能出现很大的范围，
这是保留原代码行为，不是新加入的校正。

Steger 检测、亚像素坐标、法向和加速模块继续使用 float64；只有最后的测角
算术按旧 C++ 的 float 行为执行。中心线选择不按角度大小挑选。
原先 5–25% 和 75–95% 弧长段拟合，以及相关窗口长度/窗口敏感性检查，
不用于此兼容函数。当前默认的几何方法另有端点、覆盖和窗口敏感性检查。

```python
from steger import measure_cpp_bending, NeckBendingHistory

history = NeckBendingHistory()  # 每条被跟踪精子一个实例
measurement = measure_cpp_bending(contour)
frame_angle = measurement.angle_deg
history_median = history.update(frame_angle)
```

旧 C++ 还对同一条精子的多帧角度取中位数。这里用独立历史对象实现；
不同精子不能共用历史，跟踪重置时调用 `reset()`。无效帧不追加样本。
`measure_neck()` 本身保持无跨帧状态，当前使用新几何口径；历史中位数不能代替当前帧质量标记。

旧的端部拟合方法保留为显式函数 `measure_end_direction_bending()`，用于读取或
复现历史结果。历史报告中的角度、状态及耗时不代表当前默认方法，应重新计算。

`tools/validate_cpp_angle.py` 编译独立 C++ 小程序，逐位对照同一角度数组的
10点均值和最终 float 结果。这验证测角公式与算术口径；Python 独立 Steger 与
Halcon 的中心线检测结果尚无原生库对照，不能据此声称完整输出完全等同。
