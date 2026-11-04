# Steger Python Version

以定位和测量精度为优先的 Python Steger 曲线提取实现，可直接处理灰度 ROI，并用分割 mask 筛选目标。不需要 HALCON、不提供 UI。

基于 Steger 1998 年论文的高斯导数、Hessian 亚像素定位、方向连接、线宽估计和不对称偏差校正；参照 HALCON `lines_gauss` 官方文档提供参数和属性。

**重要的精度约定：**这是独立实现，不是 HALCON 原生库的 Python 包装。默认开启额外的原始灰度截面拟合，以降低像素采样误差。合成真值测试、实际精子样本验证和与 HALCON 的输出一致性是三项不同的验证。目前不声称已经完成后两项。

## 安装

项目使用 uv 和 Python 3.12：

```bash
uv sync --extra dev
uv run steger --help
```

依赖 NumPy、SciPy 和 Pillow；开发检查额外使用 pytest。`uv.lock` 固定依赖版本。所有定位、卷积和拟合运算采用 `float64`，不自动归一化图像强度，不改变像素尺寸。

## 直接在 Python 中调用

```python
from steger import lines_gauss, measure_bending
from steger.io import read_image, write_json

image = read_image("roi.tif")
mask = read_image("neck_mask.png", mask=True)

result = lines_gauss(
    image,
    sigma=1.5,
    low=3.0,
    high=8.0,
    light_dark="dark",      # 颈部比背景暗时；否则 light
    extract_width=True,
    line_model="gaussian",  # 依据真实灰度截面选，不是所有显微图都适合高斯模型
    complete_junctions=False,
    mask=mask,
)

write_json("result.json", result.to_dict())

# 单条已确认的颈部中心线；多条曲线时先检查叠加图再选择，不能随意拼接。
if len(result.contours) == 1:
    neck = result.contours[0]
    measurement = measure_bending(neck)
    print(measurement.angle_deg)
```

参数示例不是精子图像的通用最佳参数。灰度为 `[0,1]` 时，阈值应相应调整；不能把用于 `[0,255]` 的 `Low/High` 原样使用。

## 命令行

```bash
uv run steger detect roi.tif --mask neck_mask.png \
  --sigma 1.5 --low 3 --high 8 --polarity dark --model gaussian \
  --no-complete-junctions --output result.json --overlay overlay.png
```

检测并测量单条颈部：

```bash
uv run steger detect roi.tif --mask neck_mask.png \
  --sigma 1.5 --low 3 --high 8 --polarity dark --model gaussian \
  --no-complete-junctions --measure-neck --output neck.json --overlay neck.png
```

输入支持 PNG/TIFF 等 Pillow 可读图像及 `.npy` 数组；RGB 图像在 I/O 层显式转为亮度，核心接口仅接受单通道图像。16 位和浮点灰度输入保留其原始数值范围。mask 必须与 ROI 尺寸和坐标一致；不进行自动拉伸或配准。

**mask 只限制候选中心线的位置，不把 mask 外的原图设为黑色。**卷积和截面拟合保留周围真实灰度，避免人为边界影响亚像素测量。叠加图的对比度调整只用于显示。

## 已实现的功能

| 功能 | 实现方式 |
|---|---|
| 亮线、暗线 | 最大绝对 Hessian 特征值的符号选择 |
| 高斯导数 | 论文积分形式的可分离卷积核；保留导数幅度 |
| 亚像素定位 | Taylor 定位；默认增加 Newton 迭代，处理像素中线处漏检 |
| 双阈值连接 | 方向约束、强种子与弱延续；保留闭环和交汇拓扑 |
| 线宽 | 法向两侧梯度极大值的亚像素搜索；缺失宽度沿弧长插值 |
| 不对称校正 | `bar-shaped`、`parabolic`、`gaussian` 的连续模型与数值求逆 |
| 精度增强 | 原始灰度上的像素面积积分截面拟合，可关闭 |
| 交汇补全 | 保守的端点延伸与目标曲线分割；推断点明确标记 |
| 颈部自身弯曲 | 固定弧长窗口中的正交直线拟合；直线为 0° |
| 精度验证 | 独立的连续几何图像生成、误差报告、回归测试 |

输出曲线包含 `points`（row、column）、朝曲线右侧的 `normals`、`response`、`width_left`、`width_right`，启用校正时还包含 `asymmetry`、`contrast`。`angle` 属性遵循 HALCON 文档：相对于正 row 轴逆时针计量，单位为弧度，**是法线方向，不是弯曲角度**。

`width_left/right` 是半宽。高斯模型的宽度参数是标准差，不是 FWHM；抛物线和条状模型是半支撑宽度。跨模型比较宽度时必须先统一定义。

## 质量标记

不要只读取小数坐标或角度。请同时检查：

- `width_measured` / `width_interpolated`：线宽来自观测还是插值。
- `correction_valid` / `correction_residual`：连续偏差模型求逆是否成功。
- `profile_fit_valid` / `profile_fit_rms` / `profile_fit_condition`：原始灰度拟合是否通过检查。
- `junction_completed`：点是否由交汇补全推断，不能当作真实观测测角。
- `steger_row/col`、`steger_width_left/right`：精度增强前的校正结果，便于审查算法改动。

校正或拟合失败时保留之前的可用结果，并给出失败标记，不伪造成功状态。JSON 中无法定义的数值写为 `null`。

## 精度测试与报告

```bash
uv run --extra dev pytest -q
uv run python tools/benchmark.py --output reports/accuracy.json
```

完整基准包括 84 组直线样本与一组圆弧：三种截面、多个线宽和旋转方向、非整数平移、对称/不对称背景、亮/暗线、8 位量化和固定随机噪声。图像由独立的连续模型进行像素面积积分生成，不使用阈值绘图作为亚像素真值。条状截面的像素面积为精确计算；其他截面使用 Gauss-Legendre 积分。

预设验收指标：

| 样本条件 | 中心线误差 P95 | 半宽误差 P95 | 原始灰度拟合通过率 |
|---|---:|---:|---:|
| 无噪声、模型匹配 | ≤ 0.01 px | ≤ 0.025 px | ≥ 95% |
| 对比度 150，灰度噪声标准差 1，8 位量化 | ≤ 0.08 px | ≤ 0.12 px | ≥ 95% |
| 对比度 150，灰度噪声标准差 3，8 位量化 | ≤ 0.20 px | ≤ 0.30 px | ≥ 95% |
| 指定半径圆弧 | 径向误差 P95 ≤ 0.01 px | 两个测量窗口方向差误差 ≤ 0.1° | 单独报告 |

报告保留每一组结果，包括检测失败、增强前后误差、拟合通过率和运行时间。具体实测数值见 [`reports/accuracy.json`](reports/accuracy.json)。这些指标只适用于报告中注明的合成条件，不是未知显微图像的通用误差保证。

## 接近论文流程的运行方式

```python
result = lines_gauss(
    image, sigma=1.5, low=3, high=8,
    profile_refinement=False,  # 关闭额外的原始灰度拟合
    refine=False,              # 使用论文的单次 Taylor 定位
)
```

这仍不是 HALCON 的逐位克隆。直接数值求逆、边缘插值、交汇补全和边界处理均有独立实现细节。默认模式优先降低已验证的定位误差，而不是强行复制未知的 HALCON 内部实现。

## 可选的 HALCON 对照

不影响本项目运行，也不要求安装。若将来能获得参考结果，可在有许可证的 Windows/Linux 环境运行独立脚本 `tools/export_halcon.py`，再使用：

```bash
uv run steger compare candidate.json reference.json --output comparison.json
```

比较采用双向点到线段距离，避免把不同的曲线采样间距误认为定位误差；同时报告覆盖率，避免只检测出一小段就得出“很准确”的结论。几何比较通过不代表所有属性和拓扑都已相同。

## 颈部角度的定义

默认排除首尾各 5% 弧长，在两端各取总弧长的 20% 拟合方向。两段方向统一后求偏转角，范围 `[0°,180°]`，直线为 0°。参数可调整，跨样本应保持同一规则。输出包含窗口位置、拟合残差及累计转角。

测量的是两段拟合方向之差，不等于数学曲线端点处的瞬时切线夹角。S 形曲线需要结合累计转角；累计转角对噪声和采样尺度敏感。颈部过短、曲线闭合或包含推断交汇点时会拒绝直接测角。

## 参考与范围

- [HALCON 24.11 `lines_gauss` 官方文档](https://www.mvtec.com/doc/halcon/2411/en/lines_gauss.html)
- [HALCON `get_contour_attrib_xld` 属性定义](https://www.mvtec.com/doc/halcon/2411/en/get_contour_attrib_xld.html)
- [HALCON `derivate_gauss` 官方文档](https://www.mvtec.com/doc/halcon/2411/en/derivate_gauss.html)
- C. Steger, *An Unbiased Detector of Curvilinear Structures*, TPAMI 20(2):113–125, 1998. [DOI](https://doi.org/10.1109/34.659930)
- C. Steger, *Unbiased extraction of lines with parabolic and Gaussian profiles*, CVIU 117(2):97–112, 2013. [DOI](https://doi.org/10.1016/j.cviu.2012.08.007)

详细对应关系和已知边界见 [`docs/accuracy-and-compatibility.md`](docs/accuracy-and-compatibility.md)。
