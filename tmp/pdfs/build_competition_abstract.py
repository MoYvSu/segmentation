# -*- coding: utf-8 -*-
"""Build the competition-facing two-page technical abstract."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "segmentationv2_technical_abstract.pdf"
FONT_DIR = Path(r"C:\Windows\Fonts")
pdfmetrics.registerFont(TTFont("TechSans", str(FONT_DIR / "Deng.ttf")))
pdfmetrics.registerFont(TTFont("TechSans-Bold", str(FONT_DIR / "Dengb.ttf")))

PAGE_W, PAGE_H = A4
BLUE = colors.HexColor("#173B63")
TEAL = colors.HexColor("#237D80")
PALE = colors.HexColor("#EEF5F8")
LINE = colors.HexColor("#C8D6DF")
INK = colors.HexColor("#252E36")
MUTED = colors.HexColor("#64727D")


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 15 * mm, PAGE_W - doc.rightMargin, 15 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("TechSans", 7.5)
    canvas.drawString(doc.leftMargin, 9.6 * mm, "2026AIC “AI＋钢铁”低碳钢金相图像无监督相区分割")
    canvas.drawRightString(PAGE_W - doc.rightMargin, 9.6 * mm, str(canvas.getPageNumber()))
    canvas.restoreState()


class TechnicalAbstract(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
                         topMargin=16 * mm, bottomMargin=20 * mm,
                         title="低碳钢金相图像无监督相区分割技术摘要",
                         author="segmentationv2")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                      id="main", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=draw_page)])


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TechTitle", fontName="TechSans-Bold", fontSize=20, leading=26,
                          textColor=BLUE, alignment=TA_CENTER, spaceAfter=2 * mm))
styles.add(ParagraphStyle(name="Subtitle", fontName="TechSans", fontSize=9.5, leading=13,
                          textColor=TEAL, alignment=TA_CENTER, spaceAfter=5 * mm))
styles.add(ParagraphStyle(name="H1", fontName="TechSans-Bold", fontSize=13, leading=17,
                          textColor=BLUE, spaceBefore=3 * mm, spaceAfter=2 * mm))
styles.add(ParagraphStyle(name="H2", fontName="TechSans-Bold", fontSize=10, leading=14,
                          textColor=TEAL, spaceBefore=2 * mm, spaceAfter=1 * mm))
styles.add(ParagraphStyle(name="Body", fontName="TechSans", fontSize=10, leading=15,
                          textColor=INK, spaceAfter=2.3 * mm))
styles.add(ParagraphStyle(name="Small", fontName="TechSans", fontSize=8.1, leading=11,
                          textColor=MUTED, spaceAfter=1.2 * mm))
styles.add(ParagraphStyle(name="Cell", fontName="TechSans", fontSize=8.8, leading=12.3, textColor=INK))
styles.add(ParagraphStyle(name="CellHead", fontName="TechSans-Bold", fontSize=8.8, leading=12.3, textColor=BLUE))


def P(text, style="Body"):
    return Paragraph(text, styles[style])


def make_table(rows, widths):
    data = []
    for index, row in enumerate(rows):
        data.append([P(str(value), "CellHead" if index == 0 else "Cell") for value in row])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for index in range(2, len(data), 2):
        commands.append(("BACKGROUND", (0, index), (-1, index), colors.HexColor("#FAFCFD")))
    t.setStyle(TableStyle(commands))
    return t


def build_story():
    s = [P("低碳钢金相图像无监督相区分割", "TechTitle"),
         P("技术摘要 | 2026AIC “AI＋钢铁”赛道", "Subtitle")]

    s += [P("一、数据获取与标注", "H1"),
          P("项目以赛方提供的低碳钢金相图像为唯一数据来源，覆盖不同批次、腐蚀工艺和清晰度条件。训练数据包括约 1000 张无标签图像，并从中选取 32 张进行 LabelMe 多边形标注，分别标出铁素体晶粒和连续珠光体区域。每个独立铁素体晶粒和珠光体团簇均保留独立实例编号。"),
          P("为减弱多边形边缘混色、划痕和局部噪声的影响，标注后执行 CLAHE 对比度增强、Canny 边缘检测、晶粒内部腐蚀剪裁和边界带膨胀，生成语义类别图、实例图、晶界图和训练权重。原始图像不被离线改写，训练时统一使用等比缩放、letterbox 和 reflect padding。"),
          P("同时使用通用 SAM2 对无标签训练图像生成无类别几何候选。候选只描述区域的几何连通关系，不赋予铁素体/珠光体类别；经过面积筛选、重叠去除和互斥划分后，用于补充晶界连接监督。候选内部像素对作为同实例正样本，已覆盖的不同候选之间作为负样本，未覆盖区域保持忽略。")]

    s += [P("二、模型总体结构", "H1"),
          P("模型采用共享视觉表征、语义分支和几何分支的结构。共享编码器使用通用 SAM2 Hiera 多尺度特征，并以 LoRA 适配金相图像；两个任务解码头独立初始化，分别学习类别和实例边界，避免将类别判断直接当作晶粒边界。"),
          make_table([
              ["组成", "功能"],
              ["共享编码器 + LoRA", "提取多尺度纹理、灰度、晶界和形态特征"],
              ["语义分支", "输出铁素体/珠光体概率及前景区域"],
              ["几何分支", "输出 8 个邻接关系通道，判断不同距离像素是否属于同一晶粒"],
              ["实例重建模块", "将邻接关系转换为边界障碍，再生成独立晶粒实例"],
          ], [46 * mm, 126 * mm]),
          P("八通道邻接关系以短程通道为主要边界证据，以较长距离通道补充弱边界和断裂边界；长程连接受到短程边界门控，从而降低跨晶界误连接。", "Body")]

    s += [P("三、训练流程", "H1"),
          P("训练按照“先学习域内表征，再注入语义和几何任务”的顺序进行。每个阶段具有独立输入和目标，后续阶段读取经过验证的前一阶段结果。"),
          make_table([
              ["阶段", "处理内容"],
              ["1. 表征学习", "在无标签图像上进行 LoRA 自监督学习，使共享编码器适应金相纹理、照明变化和晶粒尺度。"],
              ["2. 标签净化", "将人工多边形转换为语义、实例和边界目标，并建立有效像素权重；增强只在线执行。"],
              ["3. 双任务学习", "人工标注同时监督语义和邻接关系；无类别 SAM2 候选只进入几何损失，不进入类别损失。"],
              ["4. 联合微调", "任务头稳定后，以较小学习率微调 LoRA；语义与几何分支保持独立，避免相互覆盖。"],
          ], [35 * mm, 137 * mm]),
          P("该流程逐步注入数据表征、相区类别和晶粒几何信息，同时保留中间 checkpoint，便于定位问题来自表征、任务头还是实例重建。", "Small")]

    s += [PageBreak(), P("四、推理流程", "H1"),
          make_table([
              ["步骤", "处理逻辑"],
              ["1", "输入图像等比 letterbox，经共享编码器得到多尺度特征。"],
              ["2", "语义分支预测前景与相区概率，作为实例重建的区域约束。"],
              ["3", "几何分支预测邻接关系，短程优先并融合长程证据，形成边界概率。"],
              ["4", "对高置信边界进行封边和局部缺口修复，再以受阻分水岭切分相邻晶粒。"],
          ], [18 * mm, 154 * mm]),
          Spacer(1, 4 * mm),
          P("最终输出为与输入图像同尺寸的单通道 uint16 PNG 和同名 JSON：实例图中 0 表示背景/忽略，正数表示实例编号；JSON 记录每个实例对应的相区类别。")]
    return s


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TechnicalAbstract(str(OUTPUT)).build(build_story())
    print(OUTPUT)


if __name__ == "__main__":
    main()
