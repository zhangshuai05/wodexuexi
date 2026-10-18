# -*- coding: utf-8 -*-
"""生成《会议ASR系统-完整实现方案》Excel 文档"""
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

wb = openpyxl.Workbook()

# ---------- 样式 ----------
header_fill = PatternFill("solid", fgColor="1F4E79")
header_font = Font(name="微软雅黑", size=11, bold=True, color="FFFFFF")
phase_fill = PatternFill("solid", fgColor="DDEBF7")
phase_font = Font(name="微软雅黑", size=11, bold=True, color="1F4E79")
body_font = Font(name="微软雅黑", size=10)
wrap = Alignment(wrap_text=True, vertical="top")
thin = Side(style="thin", color="B0B0B0")
border = Border(left=thin, right=thin, top=thin, bottom=thin)

def style_header(ws, ncols, row=1):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        cell.border = border

def style_body(ws, ncols, rows):
    for r in rows:
        for c in range(1, ncols + 1):
            cell = ws.cell(row=r, column=c)
            cell.font = body_font
            cell.alignment = wrap
            cell.border = border

# ============================================================
# Sheet 1: 总流程
# ============================================================
ws1 = wb.active
ws1.title = "总流程"

headers1 = ["阶段", "步骤", "任务", "关键细节（做什么/怎么做）", "可用的技术与模型", "需要调研的点", "备注/风险"]
ws1.append(headers1)

rows1 = [
    # -------------------------------------------------- 阶段A
    ["A. 采集与预处理", "A01", "确定录制方式",
     "方案1（推荐先做）：用手机/Windows 录音机录整场会议，得到WAV/MP3文件，交给离线脚本处理。方案2：Python里用 sounddevice 边收边存。先走方案1，把【音频→文本】链路跑通，再考虑实时。",
     "sounddevice 或 pyaudio（录音）；手机自带录音机（零成本）",
     "如何把手机录的MP3自动转成 16kHz 单声道 WAV（ffmpeg 一行命令）",
     "先手动录，别一上来就做实时。实时版复杂度×3（流式VAD、增量聚类、断点续记）。"],
    ["A. 采集与预处理", "A02", "音频标准化",
     "所有模型统一要求：16kHz 采样率、单声道、float32/pcm16。用 ffmpeg 或 librosa 一键转换，脚本里集中做这一个入口。",
     "ffmpeg（转换）/ librosa（重采样）/ torchaudio",
     "MP3→WAV 转换 16k 单声道的标准命令",
     "采样率不对是 ASR 效果断崖式下降的头号原因，必须统一。40元麦克风若输出48kHz，也要转。"],
    ["A. 采集与预处理", "A03", "检查音频质量",
     "用波形图/响度检查：确认没有爆音、长时间静音、或整段都是噪底（说明麦克风没接对）。用 soundfile 读取后简单打印时长、采样率、最大幅度。",
     "soundfile / librosa / matplotlib（画波形）",
     "无需调研，动手即可",
     "会议录音常见的坑：忘了开麦、离麦克风太远导致信号微弱、空调风声大。"],
    # -------------------------------------------------- 阶段B
    ["B. 说话人分离（核心）", "B01", "语音活动检测 VAD",
     "把整段音频切成【有人说话】的短句段（0.3s~10s），每段同时给出开始/结束时间。静音段被剔除，后续所有步骤只针对有效语音段。",
     "FunASR 内置 FSMN-VAD（中文场景成熟）",
     "VAD 阈值怎么调：调太低把呼吸声当语音，调太高丢话头字",
     "VAD 是整条流水线的地基，输出质量决定后面一切。"],
    ["B. 说话人分离（核心）", "B02", "说话人嵌入提取",
     "对每个VAD段，用一个说话人模型提取一条固定长度向量（CAM++ 输出 192 维）。向量里编码的是【发声者是谁】，不随位置/音量变化——这正是您在【人移动位置】时依然稳定的原理。",
     "CAM++（阿里开源，`iic/speech_campplus_sv_zh-cn_16k-common`，ModelScope 下载）",
     "CAM++ 和 ECAPA-TDNN / ResNet34 在中文上的实际对比",
     "当前说话人分离主力模型，中文效果公认好。提取的是「声纹」，不记录方位，因此人对前后左右移动不影响。"],
    ["B. 说话人分离（核心）", "B03", "聚类分组",
     "把所有段的向量做无监督聚类：同一人的段聚到一起，得到每个段归属的说话人标签（speaker_1 / speaker_2 / …）。谱聚类 + 自适应停止准则自动决定【现场有几个人】。",
     "谱聚类（numpy/scikit-learn 自实现，约100行）或 FunASR 内置 diarization 流程",
     "聚类的自适应停止准则（AHC 距离阈值 vs 谱聚类特征间隙）",
     "方案①：用 FunASR 自带的 `auto_model.diarization()` 直接出分离结果（20行代码）。方案②：自己调 CAM++ 嵌入 + 谱聚类，控制力强。建议先用①，效果不够再上②。"],
    ["B. 说话人分离（核心）", "B04", "标签稳定化（核心后处理，回应您的担忧）",
     "场景：张三第一次说话被标 speaker_1，第二次因音色漂移被标成 speaker_6。解决：①全局信息聚类（一次聚完全场，而不是边听边分组）本身就能降低发生率；②再加一道【相邻相似合并】：如果两个非相邻段的说话人嵌入余弦相似度 > 0.7，且二者在时间上邻近，则强制并为一个标签。",
     "余弦相似度、连接分量合并（自写约30行）",
     "F1：时间上【多近算近】？F2：相似度阈值取多少？建议先在3人×5分钟的测试音频上扫一遍阈值，看标签错乱率",
     "这步是您最初担心【张三变 speaker_6】的正式解法，务必留出时间调试。宁可合并错（两人变一人）也别拆分错（一人变多人）——合并错的代价只是会议纪要里缺一个人名，拆分错的代价是同一人记录被拆成散乱多段。"],
    ["B. 说话人分离（核心）", "B05", "验证",
     "录一段3人轮流发言+1人中途走动的测试音频，人工核对标注：标签切换是否与真实说话人一致。用错误率（DER）量化：总时长中标签标错的占比。",
     "DER 指标、pyannote.metrics",
     "pyannote 的 DER 计算库用法",
     "这一步必须做。DER < 10% 算合格，10%~20% 可用，>20% 再调聚类参数。"],
    # -------------------------------------------------- 阶段C
    ["C. 语音转写", "C01", "ASR 转写",
     "对每个VAD段分别识别，得到文字 + 每段说话人标签 + 起止时间。FunASR 用 Paraformer 模型（`iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`）。",
     "FunASR / Paraformer-large（中文识别效果顶级）",
     "SenseVoice 与 Paraformer 在中文表现上的差异",
     "Paraformer 中文识别效果极强，离线批处理，且和 CAM++、VAD 同属 FunASR 生态，接口统一。"],
    ["C. 语音转写", "C02", "结果对齐与合并",
     "把 ASR 输出和阶段B的分离结果在时间轴上对齐：一段说话人段落里可能包含多句 ASR 文本，或一句话跨多个段——把最小时间粒度统一为【段】，保证最终表格里一人一句。",
     "pandas（时间轴对齐与合并）",
     "VAD 段边界与 ASR 句边界不一致时的合并策略",
     "对齐错误会造成【张三的话里掺着李四】的尴尬，这里多花点心思。"],
    # -------------------------------------------------- 阶段D
    ["D. 说话人身份推断", "D01", "声纹→姓名映射（语义层）",
     "把阶段B的输出序列（speaker_1 说了…speaker_2 说了…）整段丢给 LLM（DeepSeek），让它推断：哪段里有互相称呼（李总/张工）、哪段里有【大家好我是XXX】、哪段上有【接下来的内容由XXX讲解】——靠语义线索把 speaker_X 映射到真名。",
     "DeepSeek API（便宜，中文强）或本地部署 Qwen",
     "如何让 LLM 区分【真名字】与【口头称呼】（如老板/领导）；LLM 输出置信度与不确定处理",
     "LLM 推断适合语义层面的柔性推断，但注意：会议里可能有人全称、有人只剩【老师】。能推断出多少是惊喜，推断不出是常态。"],
    ["D. 说话人身份推断", "D02", "兜底策略：猜不出时怎么办",
     "规则①：说话人只有自我介绍或明确被指认时才赋名；规则②：未识别的说话人保持 speaker_X（或 guest_1）；规则③：若人名与说话人无法对应，也保留原标签，等后续人工补录。",
     "规则引擎（简单if-else）+ LLM 输出置信度低于阈值的自动降级",
     "暂无",
     "这就是您说【不猜，就不错】原则的落地。宁可不标名，也不标错名。"],
    ["D. 说话人身份推断", "D03", "多轮对话中的身份跟踪",
     "会议不是一锤子买卖，可能中途有人离席、有人加入、有人从未被直接称呼。需要把整场对话的所有片段（不只是相邻两句）交给 LLM 综合推断，并输出【某说话人对应真人】的置信度。",
     "LLM 多轮对话/长上下文处理；可选 RAG（把会议所有提及人名段落作为证据库）",
     "LLM 长上下文理解；跨片段推断同一人；如何量化置信度",
     "设计 LLM 的输入结构：把说话人标签序列+转写文本打包成一个 Prompt，让 LLM 统一输出映射表。这样【领导懂技术插话】也不怕——只管标注现实，不预测胡说。"],
    # -------------------------------------------------- 阶段E
    ["E. 集成与交付", "E01", "输出会议纪要",
     "最终产物：①带人名（或标注名）的逐句转写；②按发言人的统计（每人发言时长、次数）；③结构化摘要（结论、行动项、待办）。LLM 输出 Markdown 或 Word。",
     "DeepSeek（纪要生成）；python-docx 生成 Word",
     "长会议纪要生成的分段处理策略",
     "把会议转写全文丢给 LLM 时，若超长（>30分钟），需分段+摘要+再融合。"],
    ["E. 集成与交付", "E02", "端到端跑通与调优",
     "把 A→E 写成一条完整脚本：输入：会议录音（wav/mp3）→ 输出：纪要（md/docx）。再针对自己开的会反复测试，记录错误案例，回头修正。",
     "Python + ffmpeg + FunASR + DeepSeek",
     "错误案例分析、参数寻优、长期迭代机制",
     "里程碑建议：①先用单文件脚本离线跑通；②再用测试音频验证分离；③再接入 LLM 做纪要。"],
]

for r in rows1:
    ws1.append(r)
style_header(ws1, 7)
style_body(ws1, 7, range(2, ws1.max_row + 1))

# 阶段列合并着色（保持视觉分组）
phase_ranges = []
for r in range(2, ws1.max_row + 1):
    p = ws1.cell(row=r, column=1).value
    if phase_ranges and phase_ranges[-1][0] == p:
        phase_ranges[-1][2] = r
    else:
        phase_ranges.append([p, r, r])
for p, r1, r2 in phase_ranges:
    ws1.merge_cells(start_row=r1, start_column=1, end_row=r2, end_column=1)
    for rr in range(r1, r2 + 1):
        c = ws1.cell(row=rr, column=1)
        c.fill = phase_fill
        c.font = phase_font
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")

widths1 = [16, 7, 22, 52, 34, 32, 38]
for i, w in enumerate(widths1, 1):
    ws1.column_dimensions[get_column_letter(i)].width = w
ws1.freeze_panes = "A2"
ws1.row_dimensions[1].height = 30

# ============================================================
# Sheet 2: 技术选型与调研
# ============================================================
ws2 = wb.create_sheet("技术选型与调研")
headers2 = ["环节", "用途", "推荐方案（可直接装）", "权威/主流替代", "安装/下载命令", "需调研的点", "选型理由 / 注意"]
ws2.append(headers2)

rows2 = [
    ["环境", "Python 运行环境", "Python 3.10/3.11（虚拟环境）", "3.12（次选）",
     "conda create -n asr python=3.11",
     "FunASR 对 3.12 的兼容性", "3.10/3.11 库兼容最稳。建议独立环境，别动系统 Python。"],
    ["环境", "PyTorch 后端", "torch 2.x + CPU（Windows 无独显）", "torch+cu121（有独显）",
     "pip install torch --index-url https://download.pytorch.org/whl/cpu",
     "CPU 跑 5 分钟音频耗时实测（一般几分钟内能接受）", "无独显选 CPU 即可，音频任务算力要求不高。有CUDA更大，但别为这个装。"],
    ["音频", "重采样/格式转换", "ffmpeg", "librosa / torchaudio",
     "pip install ffmpeg-python；系统装 ffmpeg.exe",
     "无", "所有音频统一 16kHz 单声道 WAV，这是 FunASR 系列模型的硬性输入。"],
    ["音频", "读取与回放", "soundfile", "librosa",
     "pip install soundfile",
     "无", "读取 WAV/FLAC，无依赖，接口简单。"],
    ["分离", "VAD 说话检测", "FunASR 内置 FSMN-VAD", "Silero VAD（英语强）",
     "pip install funasr modelscope",
     "VAD 阈值对中文口音的适配", "中文会议场景，FunASR 生态一致性好。"],
    ["分离", "说话人嵌入（声纹）", "CAM++（`speech_campplus_sv_zh-cn_16k-common`）", "ECAPA-TDNN / ResNet34",
     "from modelscope import snapshot_download; snapshot_download('iic/speech_campplus_sv_zh-cn_16k-common')",
     "CAM++ vs ECAPA 中文真实效果对比", "阿里开源，中文训练，192维嵌入，当前主流。"],
    ["分离", "聚类分组", "谱聚类（自写，scikit-learn）", "FunASR 内置 diarization（AHC）",
     "pip install scikit-learn",
     "自适应确定说话人数量的准则",
     "先用 FunASR 自带的 diarization 全流程（省事）；效果不够再自写光谱聚类，可精准控制合并阈值。"],
    ["分离", "标签合并后处理", "自写：余弦相似度 + 近邻合并", "凝聚层次聚类（AHC）",
     "（纯代码，无需额外包）",
     "合并阈值 0.7 的窗口扫描（建议 0.6~0.8 各试）",
     "解决「张三变 speaker_6」的核心步骤。宁可合并错不可拆分错。"],
    ["分离", "评估指标", "DER (Diarization Error Rate)", "JER / 纯净度（纯度）",
     "pip install pyannote.metrics",
     "pyannote.metrics 与 FunASR 输出格式的对接", "DER<10% 优秀，10~20% 可用，>20% 需调参会参数。"],
    ["ASR", "转写引擎", "Paraformer-large（中文）", "SenseVoice（鲁棒）",
     "FunASR 自动拉取 `speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`",
     "SenseVoice 对带噪/口音强的会议音频的实际增益", "Paraformer 识别中文准确率顶尖；SenseVoice 抗噪更好，若测试音频噪声大可换。"],
    ["LLM", "身份推断+纪要", "DeepSeek API（便宜）", "本地 Qwen2.5 / 智谱 / 通义",
     "pip install openai（兼容接口）",
     "DeepSeek API 每次会议推断的成本（约几分钱/场）", "选 DeepSeek 很好：中文强、便宜。若走纯本地，Qwen 可离线部署，但需显存≥8GB。"],
    ["LLM", "长会议分段处理", "分段摘要 + 递归合并", "MapReduce 摘要",
     "（纯 Prompt/代码设计）",
     "会议 >30 分钟时 LLM 一次放不下的策略", "按章节拆、每段摘要、再合并，避免单次上下文爆表。"],
    ["交付", "会议纪要文档", "Markdown → Word（python-docx）", "直接 Markdown",
     "pip install python-docx",
     "无", "先用 Markdown 快速存档，后期再转 Word 排版。"],
]
for r in rows2:
    ws2.append(r)
style_header(ws2, 7)
style_body(ws2, 7, range(2, ws2.max_row + 1))
widths2 = [10, 18, 34, 26, 40, 28, 34]
for i, w in enumerate(widths2, 1):
    ws2.column_dimensions[get_column_letter(i)].width = w
ws2.freeze_panes = "A2"
ws2.row_dimensions[1].height = 30

# ============================================================
# Sheet 3: 实施排期与验证
# ============================================================
ws3 = wb.create_sheet("实施排期与验证")
headers3 = ["里程碑", "任务", "交付物", "验收标准", "预计耗时（业余）", "备注"]
ws3.append(headers3)
rows3 = [
    ["M1", "环境搭建 + 跑通 FunASR 官方 demo（识别一句话）", "能运行的演示脚本", "命令行输入 WAV 能输出文字", "0.5~1天", "这一步能跑通，说明环境无问题。"],
    ["M2", "录 3 人交替发言 + 1 人中途走动的 5 分钟测试音频", "测试音频文件", "有明确真值标签（自己用Excel记）", "0.5天", "建议录【会议风格】而非朗诵风格，才贴近真实。"],
    ["M3", "跑通 FunASR 说话人分离（VAD+CAM+++聚类）", "分离结果时间轴 + 标签", "DER < 20%；能看到 speaker_1/2/3 大体正确", "1~2天", "若出现【同一人被拆多个标签】，启用标签合并后处理（核心）"],
    ["M4", "跑通 ASR 转写，分段对齐", "带时间戳的逐句转写", "转写文字基本准确（错误率≤10%）", "0.5~1天", "对齐关注跨段句子的合并。"],
    ["M5", "接入 LLM（DeepSeek）生成会议纪要", "Markdown 会议纪要", "含：结论、行动项、待办；能输出部分人名映射", "1天", "确定 DeepSeek 的用量和成本。"],
    ["M6", "端到端集成 + 文档", "一键脚本 + 使用文档", "输入 WAV 一键出纪要，无需人工干预", "1~2天", "交付给全校同学使用。"],
    ["M7", "真实会议测试 + 迭代", "真实记录 + 错误列表", "连续 3 场会议可用", "持续", "真实会议噪声、方言、口音都会暴露，记录再迭代。"],
]
for r in rows3:
    ws3.append(r)
style_header(ws3, 7)
style_body(ws3, 7, range(2, ws3.max_row + 1))
widths3 = [10, 34, 26, 32, 16, 30]
for i, w in enumerate(widths3, 1):
    ws3.column_dimensions[get_column_letter(i)].width = w
ws3.freeze_panes = "A2"
ws3.row_dimensions[1].height = 30

# 保存
out = r"c:\Users\EDY\Desktop\new\docs\会议ASR系统_完整实现方案.xlsx"
wb.save(out)
print("saved:", out)
