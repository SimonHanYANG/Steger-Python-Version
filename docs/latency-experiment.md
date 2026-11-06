# 完整流程加速实验

2026-11-06，分支 `experiment/latency-1ms-20261106`，直接基于最新发行版
`v1.0.2`（`f2e3081`）。main、mask-only和高斯范围实验分支均保留。

## 结论

**完整平均耗时从4.596 ms降至3.253 ms，降低29.2%；本轮尚未达到1 ms。**

| 同轮测试版本 | 平均ms | 中位数ms | P95 ms | 本轮最大ms |
|---|---:|---:|---:|---:|
| 发行版v1.0.2 | 4.596 | 4.114 | 7.400 | 15.997 |
| 本次加速版 | 3.253 | 2.934 | 5.113 | 19.671 |

本次重新测量发行版作为对照，不能直接拿不同时间的旧统计计算提速比例。
平均节省1.342 ms，按整轮配对重采样得到的95%区间为1.322—1.363 ms。
新版3480次中，3279次低于5 ms、没有低于1 ms的调用；存在偶发长耗时，
不能把平均时间理解成每次调用的上限。

本机29张图的完整结果字典全部一致。8种候选平滑程度、明暗两方向下的796条中心线，
坐标、法向、响应及检测输出均与发行版一致。仍为26张有最终角度、3张没有。

没有使用mask-only，没有缩短高斯范围、删减候选平滑程度、降低分辨率或修改角度公式。
这次优先加快执行方式，保留已接受的计算规则。

## 怎样提速

1. **主进程直接与五个计算进程通信。**每个进程仍对同一张ROI尝试一种平滑程度，
   完成高斯导数、定位、连接中心线和候选测角；减少任务管理线程与结果转发。
2. **把反复运行的连接循环编译后执行。**包括去掉重复点、寻找相邻点、连接中心线和调整法向。
   点少时直接算点间距离，省去建立搜索树；接近距离、方向或评分判断边界时回到原方法。
   点多时仍使用原搜索树，避免直接计算全部点对变慢。
3. **加快方向拟合前的取点。**弧长累计、重复点判断、沿中心线取点和减去均值改成编译循环。
   后面的方向拟合、异常末端处理和测角公式保持原样。
   返回结果时，不再递归复制已经不可修改的数值和元组。
4. **只缩小二值mask的距离计算范围。**这是计算中心线离mask边缘有多远。
   mask外的距离本来就是零，保留mask矩形范围及一圈背景即可得到相同距离。
   原始灰度图、高斯导数和插值继续使用整张ROI。

编译设置仍保留严格浮点运算，没有启用fast-math或换成float32。
还试过重写高斯卷积和插值前的数据准备，但没有稳定收益，最终继续使用原SciPy实现。

## 各步骤耗时

分步测试单独运行，各步骤按同时运行的进程中最慢者计算，避免把五个进程的CPU时间相加。

| 步骤 | 平均ms |
|---|---:|
| 原图和mask整理 | 0.298 |
| 高斯导数 | 0.533 |
| 初始中心点和法向 | 0.328 |
| 亚像素精细定位 | 0.619 |
| 连接中心线 | 0.182 |
| 候选线分段拟合 | 0.492 |
| 选线及最终角度 | 0.230 |
| 数据交接与调度 | 0.392 |
| 分步测试合计 | 3.074 |

添加计时标记和不同轮次的系统负载会影响数据；分步合计对应这次分步测试的实际总时间，
不强行凑成另一次完整计时的3.253 ms。

高斯导数、精细定位和候选拟合仍占较多时间，完整计算中也包含输入整理及进程通信。
当前保留完整计算规则的这套实现尚未达到1 ms，不能把某个较快步骤当作整体耗时。

## 验证与测试范围

- 启用加速模块：203项回归测试全部通过。
- 设置 `STEGER_DISABLE_NATIVE=1`，关闭生产路径的编译加速：203项测试全部通过。
- 新增了并发调用、变化尺寸、只读输入、子进程异常退出、初始化失败和共享内存释放检查。
- 100组随机点集与冻结的发行版连接结果逐位比较；原有拓扑与阈值边界测试继续通过。
- 原有84组合成真值定位/线宽/校正验收及圆弧验收全部通过，门槛未放宽。
- 真实图像的全部8种平滑程度、明暗两方向检测结果及796条中心线与发行版逐位一致。
- 计时过程中，每次调用的完整结果均检查一致。

环境：Intel i5-12500、Windows、Python3.12.13、NumPy2.5.4、SciPy1.18.1，
本地MSVC严格浮点编译模块。29对64×64输入，每版120轮、3480次，合计6960次；
分步测试另跑60轮、1740次。

两版在同一父线程随机交替计算，均提前启动五个进程；输入已经在内存。
完整计时包括输入整理、找线、必要补试、通信、选线和测角，
不含读图、分割、绘图、进程启动及软件排队。
软件中分割等线程同时占用CPU时的延迟需要另行测试。
结果一致的结论针对本环境及验证数据，不等于所有新图和平台的普遍保证。

## 软件中怎样调用

本机的扩展已重新编译。其他环境需要从本分支构建，并重启Python/软件加载新模块：

```powershell
uv sync --frozen --extra dev --extra native
uv run --frozen --extra dev --extra native python tools/build_native.py
```

将以下流程接入软件的启动入口；五个进程只创建一次，后续不断接收已有线程提供的ROI和mask：

```python
from multiprocessing import freeze_support
from steger.worker import FastNeckAngleWorker, fast_acceleration_available

def main():
    if not fast_acceleration_available():
        raise RuntimeError("请先重新编译本分支的加速模块")
    with FastNeckAngleWorker(workers=5, max_shape=(64, 64)) as worker:
        for roi, mask in software_inputs():
            result = worker.measure(roi, mask)
            consume_result(result)

if __name__ == '__main__':
    freeze_support()
    main()
```

`software_inputs()` 和 `consume_result()` 表示你的输入线程与结果接收逻辑。
输入格式和返回字段与原来一致；ROI尺寸更大时相应设置 `max_shape`。
同一个worker的并发调用仍按顺序执行，共享一份输入空间。
没有构建可选模块时仍可使用Python/SciPy路径，但不适用这里的加速耗时。

## 复现和报告

需要本地 `neck_mask` 中的29对图片和Git中保留的发行版对象：

```powershell
$env:PYTHONUTF8='1'
uv run --frozen --extra dev --extra native pytest -q
uv run --frozen --extra dev --extra native python tools/benchmark_latency.py --rounds 120 --name paired
uv run --frozen --extra dev --extra native python tools/profile_latest_steps.py --fast --rounds 60 --output reports/neck-analysis/latency-1ms-20261106/steps.json
uv run --frozen --extra dev --extra native python tools/benchmark.py --output reports/neck-analysis/latency-1ms-20261106/truth/accuracy.json
uv run --frozen --extra dev --extra native python tools/build_latency_report.py
```

对照脚本从固定发行版Git对象构造独立Python包，验证快路径已加载。
CPython不能安全地隔离两个使用相同初始化名称的编译模块，因此两版共用原样保留的
Newton定位/采样函数；脚本检查这段源码未变化。发行版Python代码没有调用新增的编译函数。

本地报告：`reports/neck-analysis/latency-1ms-20261106/index.html`。
包含29张精子的mask轮廓、中心线、点、法向、拟合方向、角度及平均耗时对比，
所有图片内嵌，可单独分享HTML。
`paired.json`、`steps.json`、`summary.json`、`per-image.csv` 保存详细数据。
真实图片和生成报告保持本地，不上传公开仓库。
