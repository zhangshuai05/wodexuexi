import json

a = json.load(open('output/_audit_v7_v12.json', encoding='utf-8'))
b = json.load(open('output/_audit_v7_v12_variants.json', encoding='utf-8'))
for r in a:
    r['set'] = 'A'
for r in b:
    r['set'] = 'B'
allr = a + b


def tot(rows):
    tw = sum(r['turns'] for r in rows)
    v7 = sum(r['v7']['acc'] * r['turns'] for r in rows) / tw
    v12 = sum(r['v12']['acc'] * r['turns'] for r in rows) / tw
    return v7, v12, tw


print('=== 加权轮次准确率 ===')
for name, rows in [('A 主测 5 文件', a), ('B 变体 12 文件', b), ('A+B 全部 17 文件', allr)]:
    v7, v12, tw = tot(rows)
    print('%-20s turns=%3d  v7=%.3f  v12=%.3f  %+.3f' % (name, tw, v7, v12, v12 - v7))

wr = sum(1 for r in allr if r['v12']['acc'] > r['v7']['acc'])
lr = sum(1 for r in allr if r['v12']['acc'] < r['v7']['acc'])
print('\n逐文件: v12 赢 %d / v7 赢 %d / 平 %d  (共 %d)' % (wr, lr, len(allr) - wr - lr, len(allr)))

print('\n=== 人数正确 (自动K == GT人数) ===')
for name, rows in [('A', a), ('B', b), ('A+B', allr)]:
    n7 = sum(1 for r in rows if r['v7']['k'] == r['gt_spk'])
    n12 = sum(1 for r in rows if r['v12']['k'] == r['gt_spk'])
    print('  %-4s v7 %2d/%2d   v12 %2d/%2d' % (name, n7, len(rows), n12, len(rows)))

print('\n=== 逐文件明细 (A+B) ===')
print('%-34s %4s %4s %7s %7s %7s %5s %5s' % ('file', 'turn', 'gt', 'v7', 'v12', 'delta', 'K7', 'K12'))
for r in allr:
    d = r['v12']['acc'] - r['v7']['acc']
    flag = 'v12' if d > 0 else ('v7' if d < 0 else '=')
    print('%-34s %4d %4d %7.3f %7.3f %+7.3f %5s %5s  %s' % (
        r['file'][:34], r['turns'], r['gt_spk'], r['v7']['acc'], r['v12']['acc'],
        d, r['v7']['k'], r['v12']['k'], flag))

print('\n=== v12 塌成单人 / 人数不足的文件 ===')
for r in allr:
    if r['v12']['k'] < r['gt_spk']:
        print('  %-34s gt=%d v12.K=%s v7.K=%s path=%s' % (
            r['file'][:34], r['gt_spk'], r['v12']['k'], r['v7']['k'], r['v12']['path']))
