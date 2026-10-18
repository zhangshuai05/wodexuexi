import json

a = json.load(open('output/_audit_v7_v12.json', encoding='utf-8'))
b = json.load(open('output/_audit_v7_v12_variants.json', encoding='utf-8'))
allr = a + b

print('=== 按素材会话分组（A 与 B 的对应关系）===')
print('A 集的 CallMerge_X 由两段 Call 拼接；B 集的 3spk_*_X+Y 由同样两段拼接')
print('所以 A/B 是同一批录音的不同拼法，不是独立素材\n')

groups = {
    '0228+0836': [], '0454+0948': [], '0657+1031': [], '1423+1842': [], '1506+1938': [],
    'selfsplit_0228': [], 'selfsplit_1605': [],
}
for r in allr:
    f = r['file']
    if f.startswith('CallMerge_'):
        key = f.replace('CallMerge_', '').replace('_', '+')
    elif f.startswith('selfsplit'):
        key = 'selfsplit_' + f.split('_')[-1]
    else:
        key = f.split('_', 3)[-1].split('.')[0]
    if key in groups:
        groups[key].append(r)
    else:
        print('unmatched:', f, '->', key)

print('%-18s %5s %6s %6s %7s' % ('素材', '文件', 'v7', 'v12', 'delta'))
for key, rows in groups.items():
    if not rows:
        continue
    tw = sum(r['turns'] for r in rows)
    v7 = sum(r['v7']['acc'] * r['turns'] for r in rows) / tw
    v12 = sum(r['v12']['acc'] * r['turns'] for r in rows) / tw
    print('%-18s %5d %6.3f %6.3f %+7.3f' % (key, len(rows), v7, v12, v12 - v7))

print()
print('=== 求和检查 ===')
tw = sum(r['turns'] for r in allr)
print('全部 %d 文件: v7=%.3f v12=%.3f' % (
    len(allr),
    sum(r['v7']['acc'] * r['turns'] for r in allr) / tw,
    sum(r['v12']['acc'] * r['turns'] for r in allr) / tw))
