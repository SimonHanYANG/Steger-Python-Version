# 严格mask内计算实验

分支：`experiment/mask-only-20261106`，基于main提交`877bbc2`。不修改main或v1.0.2发布。

## 使用方法

```powershell
uv sync --frozen --extra dev --extra native
uv run --frozen --extra dev --extra native python tools/build_native.py
uv run --frozen --extra dev --extra native pytest -q
uv run --frozen --extra dev --extra native python tools/benchmark_mask_only.py --rounds 60
uv run --frozen --extra dev --extra native python tools/build_mask_only_report.py
```

对照接口保持原样。严格mask实验使用独立入口：

```python
from steger.mask_only import measure_neck_mask_only, MaskOnlyNeckAngleWorker

def main():
    with MaskOnlyNeckAngleWorker(workers=5) as worker:
        result = worker.measure(roi_rgb, neck_mask)
        angle = result['angle_deg']  # 可能为None

if __name__ == '__main__':
    main()
```

实验需要编译`steger._mask_native`。本次在Windows x64验证；没有提供实验内核的Python后备实现。
原有 `measure_neck`、`NeckAngleWorker` 继续作为main对照，避免混淆实验与正式结果。

## 哪些数据参与计算

- 原图仅提取最大颈部mask连通区域内的M个像素；灰度转换和有限值检查仅用于这些值。
- 高斯导数只在M个目标像素与M个源像素之间计算，忽略超出卷积支持范围的配对。
- 缺少的外部邻居不以黑色填充。对mask内高斯加权和除以有效权重总和，并用商法则计算一、二阶导数。
- 整图递归样条预滤波被替换为M个mask节点上的归一化三次插值：只对实际mask节点建立插值方程并求解，查询时不访问外部系数。
- 初始点、每次Newton修正和最终候选点均限制在mask像素覆盖的区域内。
- 二值mask的扫描、连通区域识别、距离及索引图仍有区域管理操作；它们不读取原图mask外的灰度。

边界权重和插值定义均与main不同，因此这是新检测方案，而不是逐位等价的加速。
平滑参数、双极性搜索、补救条件、六次修正上限、连接及几何测角规则保持相同。

## 本次结果

29对64×64实图，两种模式各1,740次交替完整调用，均使用5个提前建立的工作进程：

| 方案 | 平均ms | 中位数ms | P95 ms | 有最终角度 |
|---|---:|---:|---:|---:|
| main | 5.93 | 5.20 | 10.17 | 26/29 |
| 严格mask内 | 8.54 | 7.66 | 14.32 | 16/29 |

本实现平均慢43.9%，少10张输出角度。双方有角度的16张，角度变化绝对值平均5.21°、最大14.86°。
这些是与已确认版本的差异，不是人工真值误差。当前不适合替代main。

完整耗时包含输入处理、搜索、补救、测角及通信，不含读图、分割、绘图、启动和排队；包含没有角度的图片。
不能拿前一天另一次测量的5.59ms与本轮实验直接计算提速比例。

## 验证与本地报告

197项测试通过，其中6项专门验证mask隔离、独立数值公式、常量边界、插值节点、进程一致性和点的位置。
29张图的mask外灰度改为NaN或无穷大后，实验版完整输出不变；所有候选点仍在mask内。
main的29张完整输出仍与已确认版本一致。

本地报告：`reports/neck-analysis/mask-only-20261106/index.html`，内嵌全部29张对比图，可独立分享。
另有原始计时`benchmark.json`、逐图CSV和PNG。原图与报告继续排除在Git之外。
