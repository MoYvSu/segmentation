# -*- coding: utf-8 -*-
"""Build the <=2-page competition technical abstract PDF."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
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
OUTPUT = ROOT / "output" / "pdf" / "segmentationv2_technical_abstract.pdf"
FIGURE = ROOT / "downloads" / "e10a_analysis" / "e10a_visual_compare" / "overview.png"
FONT_DIR = Path(r"C:\Windows\Fonts")
pdfmetrics.registerFont(TTFont("AbstractSans", str(FONT_DIR / "Deng.ttf")))
pdfmetrics.registerFont(TTFont("AbstractSans-Bold", str(FONT_DIR / "Dengb.ttf")))

PAGE_W, PAGE_H = A4
BLUE = colors.HexColor("#173B63")
TEAL = colors.HexColor("#237D80")
PALE = colors.HexColor("#EAF2F8")
INK = colors.HexColor("#202A33")
MUTED = colors.HexColor("#5D6B78")


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7E0E8"))
    canvas.setLineWidth(0.45)
    canvas.line(doc.leftMargin, 15 * mm, PAGE_W - doc.rightMargin, 15 * mm)
    canvas.setFont("AbstractSans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 9.7 * mm, "2026AIC “AI＋钢铁”低碳钢金相图像无监督相区分割")
    canvas.drawRightString(PAGE_W - doc.rightMargin, 9.7 * mm, str(canvas.getPageNumber()))
    canvas.restoreState()


class AbstractDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                         topMargin=14 * mm, bottomMargin=19 * mm,
                         title="低碳钢金相图像无监督相区分割技术摘要",
                         author="segmentationv2")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                      id="abstract", leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="abstract", frames=frame, onPage=on_page)])


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleX", fontName="AbstractSans-Bold", fontSize=19,
                          leading=24, textColor=BLUE, alignment=TA_CENTER, spaceAfter=2 * mm))
styles.add(ParagraphStyle(name="SubX", fontName="AbstractSans", fontSize=9.5,
                          leading=13, textColor=TEAL, alignment=TA_CENTER, spaceAfter=3 * mm))
styles.add(ParagraphStyle(name="H1X", fontName="AbstractSans-Bold", fontSize=12.5,
                          leading=16, textColor=BLUE, spaceBefore=2.5 * mm, spaceAfter=1.5 * mm))
styles.add(ParagraphStyle(name="H2X", fontName="AbstractSans-Bold", fontSize=9.5,
                          leading=13, textColor=TEAL, spaceBefore=1.5 * mm, spaceAfter=1 * mm))
styles.add(ParagraphStyle(name="BodyX", fontName="AbstractSans", fontSize=8.7,
                          leading=12.4, textColor=INK, spaceAfter=1.5 * mm))
styles.add(ParagraphStyle(name="SmallX", fontName="AbstractSans", fontSize=7.2,
                          leading=9.5, textColor=MUTED, spaceAfter=1 * mm))
styles.add(ParagraphStyle(name="CellX", fontName="AbstractSans", fontSize=7.35,
                          leading=9.3, textColor=INK))
styles.add(ParagraphStyle(name="CellHeadX", fontName="AbstractSans-Bold", fontSize=7.35,
                          leading=9.3, textColor=BLUE))
styles.add(ParagraphStyle(name="CalloutX", fontName="AbstractSans", fontSize=8.5,
                          leading=12, textColor=BLUE, backColor=PALE, borderColor=colors.HexColor("#BCD0E1"),
                          borderWidth=0.6, borderPadding=3 * mm, spaceBefore=1 * mm, spaceAfter=2 * mm))


def P(text, style="BodyX"):
    return Paragraph(text, styles[style])


def make_table(rows, widths, header=True):
    converted = []
    for r, row in enumerate(rows):
        converted.append([c if hasattr(c, "wrap") else P(str(c), "CellHeadX" if header and r == 0 else "CellX") for c in row])
    t = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C8D3DD")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), PALE))
        for i in range(2, len(converted), 2):
            cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F7FAFC")))
    t.setStyle(TableStyle(cmds))
    return t


def story():
    s = [Spacer(1, 1 * mm), P("低碳钢金相图像无监督相区分割", "TitleX"),
         P("技术摘要 | 2026AIC “AI＋钢铁”赛道 | 2026-09-06", "SubX")]
    s += [P("摘要", "H1X"), P("本项目针对低碳钢铁素体-珠光体金相图像，提出一种面向实例分割的无监督/半监督方案。方法以通用 SAM2 Hiera 表征和域内 LoRA 自监督学习为基础，将相区语义识别与晶粒几何切分拆成两个相互独立的任务头：semantic head 预测铁素体/珠光体，8 通道 affinity head 预测多尺度邻接像素是否属于同一晶粒。推理时融合 affinity 边界证据、封边和受阻分水岭生成实例，再以实例内语义概率投票确定类别。"),
         P("固定完整部署链的当前最佳为 E10a semantic + G4b affinity，官方黑盒结果为实例 mIoU 0.8381、铁素体平均面积项 0.8408、综合分 83.94。项目同时保留 SSL 直达双头短链作为课程学习消融，但未以未验证结果替换主线。", "CalloutX")]

    s += [P("1. 任务约束与数据", "H1X"),
          P("训练使用赛方无标签图像和合规自主标注数据；当前仓库有 32 张 LabelMe 标注图、约 1000 张无标签图及 68 张测试图部署入口。类别编码为珠光体 0、铁素体 1。标注经过 CLAHE、Canny、内部腐蚀和边界带膨胀，得到较少混色和划痕影响的语义/边界目标。输入统一采用等比 letterbox + reflect padding，保持晶粒比例。"),
          make_table([
              ["项目", "实现口径"],
              ["提交格式", "同尺寸单通道 uint16 PNG；0 为背景，1-65535 为实例 ID；另附类别 JSON"],
              ["评分", "实例 mIoU 50 分 + 铁素体平均面积 50 分；半决赛客观分 70%，答辩主观分 30%"],
              ["合规", "仅使用赛方图像和自主标注；不使用外部金相数据、测试集标签或测试集人工标注"],
              ["规模", "融合模型约 81.67M 参数，小于 500M；任务 decoder 随机初始化"],
          ], [32 * mm, 144 * mm]),
          P("官网正式“分割标签/提交结果格式”和“验证方式”要求 uint16/65535；同页“精简标注说明”残留的 1-255 属旧文案，本项目按正式提交校验口径实现。", "SmallX")]

    s += [P("2. 算法核心逻辑", "H1X"),
          make_table([
              ["模块", "核心逻辑"],
              ["共享表征", "SAM2 Hiera 多尺度特征 + LoRA；SSL 阶段从无标签图像学习域内纹理与形态"],
              ["语义分支", "E10a 随机高分辨率 semantic FPN/head；输出前景和类别概率，避免沿用旧语义决策"],
              ["几何分支", "G4b 8 通道 affinity；短程通道提供主边界证据，distance-2/4 通道补充弱边界"],
              ["无类别伪监督", "SAM2 mask 内为 positive，已覆盖 mask 间为 negative；未覆盖 pair 保持 ignore"],
              ["部署后处理", "gated fusion -> high=0.65 -> seal2/局部重建 -> 受阻分水岭 -> probability-mean 投票"],
          ], [32 * mm, 144 * mm]),
          P("训练采用分阶段信息注入：LoRA SSL -> Stage1/joint-v3 -> V6 语义锚点；E10a 独立冷启动语义头；G0/G1/G2/G4b 独立学习 affinity 几何，最后只在部署阶段组合。该结构保留了可回退 checkpoint，并能分别审计语义、几何和后处理误差。")]
    s += [PageBreak()]

    s += [P("3. 训练与推理流程", "H1X"),
          make_table([
              ["阶段", "输入与输出", "设计目的"],
              ["表征学习", "1000 张无标签图 -> LoRA SSL", "先获得域内视觉表征，不引入实例类别偏差"],
              ["监督净化", "LabelMe -> purified GT", "减少多边形混色、划痕和边界外溢"],
              ["双任务学习", "人工实例 + 无类别 SAM2 geometry", "语义和晶界连接分别注入，避免相互覆盖"],
              ["完整部署", "semantic + affinity -> instance/class", "用竞赛最终输出而非训练 loss 选择 checkpoint"],
          ], [30 * mm, 68 * mm, 78 * mm]),
          P("推理首先由 semantic 分支提供前景门控，由 affinity 融合得到边界概率。高置信边界作为受阻分水岭的硬障碍；seal2 和局部重建只修复稳定的局部缺口，不按照预测实例数或铁素体平均面积反向拟合。最终实例图使用 uint16 写出，打包器检查尺寸、单通道、dtype、ID 与 JSON 一致性。")]

    s += [P("4. 实验结果与分析", "H1X"),
          make_table([
              ["方案", "实例 mIoU", "铁素体面积项", "综合分", "判定"],
              ["V6 + G4b", "0.8441", "0.7693", "80.67", "历史基线"],
              ["E9 + G4b", "0.8421", "0.7917", "81.69", "历史回退"],
              ["E10a + G4b", "0.8381", "0.8408", "83.94", "当前主线"],
              ["graph-v1", "0.8268", "0.8365", "83.17", "未晋级"],
          ], [50 * mm, 28 * mm, 34 * mm, 25 * mm, 39 * mm]),
          P("E10a 相比 E9 的实例 mIoU 下降 0.0040，但面积项提高 0.0491，综合分提升约 2.25，说明实例欠分割/过分割对面积指标具有显著联动。G4b 的未覆盖带负 affinity 监督可减少部分合并，但会造成 affinity 标定下移，因此固定 high=0.65。graph-v2 出现不自然的笔直归并边界，G7 在固定协议下进一步减少实例并加重欠分割，均停止晋级。")]

    if FIGURE.is_file():
        fig = [P("图 1. 测试图无标签部署目检：原图 / G4b / E10a", "H2X"),
               Image(str(FIGURE), width=178 * mm, height=178 * mm * 639 / 1800),
               P("仅用于观察边界和类别颜色，不包含测试集真实标签，不替代官方评分。", "SmallX")]
        s += [KeepTogether(fig)]

    s += [P("5. 合规性、局限与结论", "H1X"),
          P("冷启动入口 repro/train.py 已通过语法、配置、数据数量、路径契约和完整阶段 dry-run 检查；历史 LoRA SSL、Stage1-LoRA、joint-v3 和 V6 checkpoint 已找回。正式 GPU 冷启动训练和完整推理复现包装不在本摘要中重新执行。"),
          P("当前主要风险是模糊、弱腐蚀和低对比区域的欠分割。强 blur/downsample 会诱发“模糊≈珠光体”的语义漂移；颜色先验虽有帮助，但不能替代语义头。后续优先等待复赛数据分布，改善共享表征和监督可靠性，不针对单张测试图继续堆叠规则。"),
          Spacer(1, 1.5 * mm),
          P("结论：E10a + G4b 是当前经过官方黑盒验证的最可靠单主干方案；SSL 直达语义/affinity 双头方案具有研究价值，但必须在同一部署链和完整竞赛指标下超过 83.94 才能替换主线。", "CalloutX"),
          P("依据：README.md、docs/PIPELINE.md、COMPETITION_RULES.md、docs/SEMANTIC_EXPERIMENT_E10A_20260828.md、docs/DIRECT_SSL_SEMANTIC_AFFINITY.md、repro/README.md；官网规则页核对日期 2026-09-04。", "SmallX")]
    return s


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AbstractDoc(str(OUTPUT)).build(story())
    print(OUTPUT)


if __name__ == "__main__":
    main()
