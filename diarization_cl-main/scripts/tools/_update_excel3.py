# -*- coding: utf-8 -*-
"""Excel：在 B06 备注中补充【已验证】的多设备融合结果；E01 备注加融合输出。"""
import openpyxl
P = r"c:\Users\EDY\Desktop\new\docs\会议ASR系统_完整实现方案.xlsx"
wb = openpyxl.load_workbook(P)

ws = wb["总流程"]
for row in ws.iter_rows(min_row=2):
    if row[1].value == "B06":
        row[6].value = (row[6].value or "") + (
            "\n【已验证 2026-08-28】双设备软融合在2m同场景对(116顾客麦+118店员麦)成功：融合后21句，"
            "顾客7句+店员14句，与人工真值基本逐句对上；2m远处顾客被还原。方法见 multi_device_fusion.py，"
            "结果见 diar_result/fusion/RESULT.md。局限：软融合按声道来源定说话人，无法处理两人同时大声。")
    if row[1].value == "E01":
        row[6].value = (row[6].value or "") + (
            " 多设备场景：可在融合结果（multi_device_fusion.py 输出）上生成纪要，说话人=声道来源。")

wb.save(P)
print("saved")
