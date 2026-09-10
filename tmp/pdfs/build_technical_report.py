# -*- coding: utf-8 -*-
"""Build the competition technical report PDF."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "segmentationv2_technical_report.pdf"
FIGURE = ROOT / "downloads" / "e10a_analysis" / "e10a_visual_compare" / "overview.png"
FONT_DIR = Path(r"C:\Windows\Fonts")

pdfmetrics.registerFont(TTFont("ReportSans", str(FONT_DIR / "Deng.ttf")))
pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(FONT_DIR / "Dengb.ttf")))


PAGE_W, PAGE_H = A4
BLUE = colors.HexColor("#173B63")
TEAL = colors.HexColor("#2E7D82")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_TEAL = colors.HexColor("#E8F4F2")
INK = colors.HexColor("#202A33")
MUTED = colors.HexColor("#5D6B78")


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates([PageTemplate(id="report", frames=frame, onPage=draw_page)])


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7E0E8"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 16 * mm, PAGE_W - doc.rightMargin, 16 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("ReportSans", 8)
    canvas.drawString(doc.leftMargin, 10.5 * mm, "低碳钢金相图像无监督相区分割 | 技术报告")
    canvas.drawRightString(PAGE_W - doc.rightMargin, 10.5 * mm, f"{canvas.getPageNumber()}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CoverTitle", fontName="ReportSans-Bold", fontSize=25, leading=33,
    textColor=BLUE, alignment=TA_CENTER, spaceAfter=8 * mm,
))
styles.add(ParagraphStyle(
    name="CoverSub", fontName="ReportSans", fontSize=13, leading=20,
    textColor=TEAL, alignment=TA_CENTER, spaceAfter=5 * mm,
))
styles.add(ParagraphStyle(
    name="H1x", fontName="ReportSans-Bold", fontSize=16, leading=22,
    textColor=BLUE, spaceBefore=5 * mm, spaceAfter=3 * mm,
))
styles.add(ParagraphStyle(
    name="H2x", fontName="ReportSans-Bold", fontSize=11.5, leading=16,
    textColor=TEAL, spaceBefore=3 * mm, spaceAfter=2 * mm,
))
styles.add(ParagraphStyle(
    name="Bodyx", fontName="ReportSans", fontSize=9.5, leading=15,
    textColor=INK, spaceAfter=2.2 * mm,
))
styles.add(ParagraphStyle(
    name="Smallx", fontName="ReportSans", fontSize=8, leading=12,
    textColor=MUTED, spaceAfter=1.5 * mm,
))
styles.add(ParagraphStyle(
    name="Callout", fontName="ReportSans", fontSize=10, leading=16,
    textColor=BLUE, backColor=PALE_BLUE, borderColor=colors.HexColor("#BCD0E1"),
    borderWidth=0.7, borderPadding=5 * mm, spaceBefore=2 * mm, spaceAfter=4 * mm,
))
styles.add(ParagraphStyle(
    name="Cell", fontName="ReportSans", fontSize=8.2, leading=11,
    textColor=INK,
))
styles.add(ParagraphStyle(
    name="CellBold", fontName="ReportSans-Bold", fontSize=8.2, leading=11,
    textColor=INK,
))


def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def bullet(text):
    return P(f"<font color='#2E7D82'>•</font> {text}", "Bodyx")


def table(data, widths, header=True, background=PALE_BLUE):
    converted = []
    for row in data:
        converted.append([
            cell if hasattr(cell, "wrap") else P(str(cell), "CellBold" if header and not converted else "Cell")
            for cell in row
        ])
    t = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D3DD")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), background),
            ("TEXTCOLOR", (0, 0), (-1, 0), BLUE),
        ])
        for i in range(1, len(converted)):
            if i % 2 == 0:
                commands.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F7FAFC")))
    t.setStyle(TableStyle(commands))
    return t


def section(title, body):
    return [P(title, "H1x")] + body


def build_story():
    story = []
    story += [Spacer(1, 28 * mm), P("低碳钢金相图像\n无监督相区分割", "CoverTitle")]
    story += [P("技术报告", "CoverSub"), Spacer(1, 7 * mm)]
    story += [P("2026AIC “AI＋钢铁”赛道 | segmentationv2", "CoverSub")]
    story += [Spacer(1, 22 * mm)]
    story += [P("报告范围：数据合规、模型结构、课程式训练、部署链路、实验结果与复现方法", "Callout")]
    story += [Spacer(1, 33 * mm), P("版本日期：2026-09-06", "Smallx")]
    story += [PageBreak()]

    story += section("摘要", [
        P("本项目面向低碳钢铁素体-珠光体金相图像，构建只依赖赛方图像和通用 SAM2 初始化的无监督/半监督实例分割系统。系统以共享的 SAM2 Hiera 编码器和 LoRA 表示为基础，采用相互独立的语义头与八通道 affinity 几何头，再通过边界融合和受阻分水岭生成实例，最后以实例级语义投票确定铁素体或珠光体类别。"),
        P("在固定的完整部署链上，当前最佳为 E10a 语义解码器 + G4b affinity + high=0.65 + seal2/局部重建 + 受阻分水岭，官方黑盒结果为实例 mIoU 0.8381、铁素体平均面积项 0.8408、综合分 83.94。该结果取代 E9 成为当前单模型主线。"),
        P("项目的主要技术判断是：实例分割中的语义识别、晶界几何和输出后处理具有不同的误差来源，不能用一个共享辅助头或连续模型融合简单替代。课程式训练将表征学习、语义监督和几何监督分开注入，再在部署阶段组合，能够保留可回退锚点并支持单变量审计。"),
        P("本报告中的官方黑盒分数、六图/无标签 monitor 代理、Oracle graph 指标和目检结论严格分开；后者仅用于诊断，不作为竞赛成绩或模型晋级依据。", "Callout"),
    ])

    story += section("1. 任务定义与竞赛约束", [
        P("任务要求对每个独立铁素体晶粒和连续珠光体团簇进行实例分割，同时输出实例索引图和实例类别 JSON。类别编码固定为：珠光体 0，铁素体 1。当前官网正式提交校验要求实例索引为与输入同尺寸的单通道 uint16 PNG，像素范围 0-65535，0 为背景/忽略，1-65535 为实例编号；本项目的打包工具按该正式格式校验。"),
        table([
            ["约束", "项目采用的实现口径"],
            ["训练数据", "赛方 1000 张无标签图像；自主标注不超过无标签数据的 50%，即 500 张"],
            ["输出", "{basename}_inst.png + {basename}_class.json；单通道 uint16，实例 ID 1-65535"],
            ["评分", "实例 mIoU 50 分 + 铁素体平均面积 50 分；半决赛客观分按 70% 折算，答辩主观分 30%"],
            ["纯无监督加分", "综合得分不低于 70 时额外 5 分，总分不超过 100"],
            ["模型规模", "总参数量小于 500M；不加载 SAM2 原生 Mask Decoder 权重"],
            ["数据边界", "不使用外部金相数据、测试集标签或测试集人工标注"],
        ], [34 * mm, 130 * mm]),
        P("官网同页的“精简标注说明”仍保留 1-255 的旧文案，但正式“分割标签/提交结果格式”和“验证方式”明确要求 uint16/65535。本项目按直接约束提交物和验证器的正式段落执行。", "Smallx"),
    ])

    story += section("2. 数据与标签构建", [
        P("当前仓库保留 32 张 LabelMe 标注图、约 1000 张赛方无标签图以及 68 张初赛测试图的部署与审计入口。标注数据不直接作为粗糙边界监督，而是经过净化形成更适合训练的语义和边界目标。"),
        P("净化流程为 CLAHE 增强、Canny 边缘检测、晶粒内部腐蚀剪裁和边界带膨胀，产出二值语义掩码、二值晶界掩码及对应权重。这样做的目的不是修改原始图像，而是减少多边形边缘混色、划痕和标注外溢对边界学习的影响。在线输入统一采用等比 letterbox 和 reflect padding，禁止挤压图像比例。"),
        P("无类别 SAM2 几何候选只用于 affinity/边界几何监督，不提供铁素体/珠光体类别。候选经过面积、重叠和互斥仲裁；同一 mask 内的像素对作为 positive，不同已覆盖 mask 之间作为 negative，covered-uncovered 和 uncovered-uncovered 保持 ignore，避免把 SAM2 漏检误写为真实边界。"),
        table([
            ["数据源", "监督内容", "用途"],
            ["LabelMe + purified GT", "类别语义、实例图、人工 affinity pair", "语义头和 affinity 头的核心监督"],
            ["SAM2 无类别候选", "仅几何 affinity pair，无 class_label", "补充晶界几何覆盖，不能改变语义类别"],
            ["无标签图像", "一致性/蒸馏与部署 monitor", "表征泛化和质量诊断，不生成测试标签"],
        ], [42 * mm, 68 * mm, 54 * mm]),
    ])

    story += section("3. 模型结构", [
        P("最终部署包将共享编码器和两个任务分支组合为一个模型，参数量约 81.67M，显著低于 500M 约束。编码器采用 SAM2 Hiera Base+ 通用视觉表征并配合 LoRA；任务解码器独立随机初始化，不加载 SAM2 原生 Mask Decoder。"),
        table([
            ["模块", "作用", "训练/部署状态"],
            ["SAM2 Hiera + LoRA", "提取多尺度纹理、晶界和形态表示", "SSL 阶段学习 LoRA；后续按阶段冻结或小学习率联合"],
            ["Semantic FPN/head", "输出铁素体概率与珠光体互补语义", "E10a 冷启动完整高分辨率语义解码器"],
            ["8-channel affinity head", "预测多尺度邻接像素是否属于同一实例", "G4b 训练；四个短程 + 两组长程通道"],
            ["Fusion + watershed", "把 affinity 转为边界障碍并切分实例", "固定部署后处理，受控封边和局部重建"],
        ], [39 * mm, 70 * mm, 55 * mm]),
        P("八通道 affinity 的核心优点是把“同一晶粒内部的连接”和“跨晶界的断开”建模为局部关系，而不是直接要求网络一次输出完整实例 ID。短程通道提供主要边界证据，distance-2/4 通道用于补充断裂和弱边界，但在部署时受到短程边界门控，降低跨晶界长距离泄漏。"),
    ])

    story += section("4. 课程式训练链路", [
        P("完整复现链由 repro/train.py 编排，阶段顺序不是无意义的重复训练，而是逐步注入不同可靠性和尺度的信息："),
        table([
            ["阶段", "主要输入", "核心信息"],
            ["LoRA SSL", "1000 张无标签图", "通用金相表征和域内纹理"],
            ["Stage1-LoRA", "净化标签 + LoRA", "初始语义/边界监督"],
            ["joint-v3 -> V6", "人工标签、半监督流、边界缓存", "稳定的语义锚点和边界基线"],
            ["E10a", "冻结 V6 特征，随机语义头", "直接优化最终语义类别，避免旧决策稀释"],
            ["G0/G1/G2/G4b", "人工 affinity + 无类别 SAM2 几何", "逐步提高晶界连接监督，独立于语义头"],
            ["部署融合", "E10a semantic + G4b affinity", "统一输出实例图和类别 JSON"],
        ], [35 * mm, 65 * mm, 64 * mm]),
        P("为检验“是否可以缩短训练链”的猜想，项目另设 SSL -> 语义/affinity 双头的 direct 方案：先 5 epoch 冻结 LoRA 训练两个随机 head，再以小学习率联合 LoRA 训练 20 epoch，并可加入无类别 SAM2 affinity 监督。该方案作为结构性对照保留，但在当前报告数据中尚未取得超过 83.94 的官方黑盒结果，因此不替换主线。"),
    ])

    story += [PageBreak()]
    story += section("5. 推理与提交链路", [
        P("最终部署遵循固定、可审计的单向流程："),
        table([
            ["步骤", "处理"],
            ["1", "输入图像等比 letterbox；共享 SAM2+LoRA 提取多尺度特征"],
            ["2", "E10a semantic head 输出前景/类别概率；G4b 输出 8 通道 affinity"],
            ["3", "短程主导的 gated fusion 形成边界概率，使用 high=0.65"],
            ["4", "seal2 与局部边界重建修复边界缺口，但不引入实例数量或平均面积规则"],
            ["5", "受阻分水岭生成实例；面积过滤和局部邻接合并只作为格式上限保护"],
            ["6", "对每个实例做 probability-mean 语义投票，写出 uint16 实例 PNG 与类别 JSON"],
            ["7", "package_submission.py 检查尺寸、单通道、uint16、ID/JSON 一致性并生成扁平 ZIP"],
        ], [17 * mm, 147 * mm]),
        P("实例级后处理不强制固定实例数，也不根据测试集统计拟合平均面积。更新后的实现允许正常使用 256 号及以上实例，只有真正超过 65535 时才按局部邻接关系合并，避免旧的 8-bit 溢出桶制造跨区域伪实例。"),
        P("提交文件契约示例：test_001_inst.png 为单通道 uint16；test_001_class.json 的键集合必须等于实例图中的非零 ID 集合，类别值只能是 0 或 1。", "Callout"),
    ])

    story += section("6. 实验结果与选择依据", [
        P("下表均为完整部署链上的官方黑盒结果或已记录的固定协议结果。主线选择严格依据实例 mIoU、铁素体平均面积项和综合分，不使用训练 loss、实例总数或 Oracle graph 指标替代。"),
        table([
            ["方案", "实例 mIoU", "铁素体面积项", "综合分", "结论"],
            ["V6 + G4b watershed", "0.8441", "0.7693", "80.67", "历史几何/语义基线"],
            ["E9 + G4b watershed", "0.8421", "0.7917", "81.69", "稳定历史回退"],
            ["E10a + G4b watershed", "0.8381", "0.8408", "83.94", "当前主线"],
            ["graph-v1 area200", "0.8268", "0.8365", "83.17", "未超过主线"],
        ], [49 * mm, 28 * mm, 32 * mm, 25 * mm, 30 * mm]),
        P("E10a 的实例 mIoU 比 E9 低 0.0040，但铁素体面积项提高 0.0491，综合分净增约 2.25。这个结果说明竞赛目标并非只追求语义像素平均精度，还必须控制实例欠分割/过分割对面积项的联动影响。"),
        P("G4b 的人工未覆盖带负 affinity 监督能够减少一部分合并，但训练后 affinity 标定整体下移；若沿用旧阈值会引入过分割，因此最终固定 high=0.65。graph-v2 出现不自然的笔直合并边界，G7 在固定协议下进一步减少实例并加重欠分割风险，均停止晋级。"),
    ])

    if FIGURE.is_file():
        figure_block = [Spacer(1, 3 * mm), P("图 1. 固定测试图上的无标签部署目检（原图 / G4b / E10a）", "H2x")]
        img = Image(str(FIGURE), width=178 * mm, height=178 * mm * 639 / 1800)
        figure_block += [img, P("该图仅用于观察实例边界和类别颜色的一致性，不包含测试集真实标签，不能替代官方黑盒评分。", "Smallx")]
        story += [KeepTogether(figure_block)]

    story += section("7. 误差分析与局限", [
        P("当前测试目检的首要问题仍是欠分割，尤其出现在低清晰度、模糊、弱腐蚀和边界对比度不足的区域。清晰区域通常能保持较好的晶粒分界；模糊区域中，affinity 的长程连接和插值后的边界带可能把多个晶粒连成大块，或让边界证据变得过宽。"),
        P("物理外观增强的实验说明，强度过大的 blur/downsample 会使模型把“模糊/低对比”错误关联到珠光体，造成语义漂移。因而增强必须保留干净样本、限制退化幅度，并通过固定部署链验证，而不能只依据训练 loss 下降。"),
        P("颜色分析显示 Lab L* 在标注内部具有较强分离，但不同图像的最佳阈值存在变化，固定颜色阈值不能替代语义头。中心热图辅助任务因与边界 FPN 共享表示而出现背景雾化、铁素体大块欠分割和珠光体碎裂，已降级为负面对照。"),
        table([
            ["风险", "当前处理", "仍需改进"],
            ["模糊区域欠分割", "高置信边界硬障碍、seal2、局部重建", "等待复赛分布，优先从表征和训练目标改善"],
            ["affinity 标定漂移", "固定 high=0.65，保留 G4b 配置快照", "单变量校准，避免大范围规则堆叠"],
            ["语义/几何互相污染", "独立任务头、冻结锚点、零残差审计", "进一步验证 direct 双头是否能稳定泛化"],
            ["测试集不可监督", "只做无标签目检和格式检查", "以复赛/半决赛官方评分作为最终证据"],
        ], [40 * mm, 63 * mm, 64 * mm]),
    ])

    story += [PageBreak()]
    story += section("8. 复现、合规与交付", [
        P("冷启动复现入口为 repro/train.py。它会检查数据数量、权重、配置和路径，按固定阶段生成 purified GT、LoRA SSL、Stage1、joint-v3、V6、E10a 和 G4b 所需产物；任何已存在的阶段产物默认不会被静默覆盖。"),
        table([
            ["检查项", "状态"],
            ["Python 语法与配置加载", "已通过"],
            ["32 张标注图、1000 张无标签图契约", "已通过"],
            ["完整阶段 dry-run", "已通过"],
            ["正式 GPU 冷启动训练", "本报告未执行；历史 checkpoint 已找回并可作校验锚点"],
            ["推理复现包装", "接口已固定，完整 repro inference 尚待实现"],
        ], [68 * mm, 99 * mm]),
        P("关键历史资产包括 LoRA SSL、Stage1-LoRA、joint-v3 和 V6 checkpoint；配置文件使用项目相对路径，不写入服务器绝对路径。统一部署包 e10a_g4b_fused.pth 将共享 encoder、E10a semantic decoder 与 G4b affinity decoder 合并为单模型，参数量约 81.67M。"),
        P("本项目不把未审核的 SAM2 候选自动送入有监督训练；候选的 class_label 必须为空，未覆盖区域必须保持 ignore。提交前使用 package_submission.py 做最终格式验证，并保存 checkpoint SHA-256、配置快照、epoch 和输出 manifest，便于组委会复现和答辩说明。", "Callout"),
    ])

    story += section("9. 结论", [
        P("当前最可靠的方案是 E10a 语义和 G4b affinity 的单主干部署，而不是把 E9、G7、graph 或多个语义 challenger 连续融合。其核心依据是：E10a 在官方指标中显著改善铁素体平均面积项；G4b 提供相对稳定的实例几何；两者可在统一部署链上逐项审计。"),
        P("短链 SSL -> 语义/affinity 双头训练具有明确研究价值：它可检验课程式阶段是否造成知识稀释，并减少多次 decoder 重置和目标切换。但短链方案必须在相同部署后处理、相同数据划分和完整官方指标下超过 83.94 才能替换主线；在此之前，它应作为结构性消融和报告中的方法学对照。"),
        P("后续优先级是等待复赛数据分布，围绕模糊和低对比工况改善共享表征、监督可靠性和轻量后处理泛化，而不是针对单张测试图继续堆叠规则。", "Callout"),
        P("参考依据：README.md；docs/PIPELINE.md；COMPETITION_RULES.md；docs/SEMANTIC_EXPERIMENT_E10A_20260828.md；docs/DIRECT_SSL_SEMANTIC_AFFINITY.md；repro/README.md；官网规则页（核对日期 2026-09-04）。", "Smallx"),
    ])

    return story


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=21 * mm,
        title="低碳钢金相图像无监督相区分割技术报告",
        author="segmentationv2",
    )
    doc.build(build_story())
    print(OUTPUT)


if __name__ == "__main__":
    main()
