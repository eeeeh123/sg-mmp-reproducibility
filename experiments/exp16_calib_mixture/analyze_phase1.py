
import json

with open('results/task_results_full.jsonl') as f:
    lines = f.readlines()

gptq_mix = {}
config_b_mix = {}

for line in lines:
    d = json.loads(line)
    m = d.get('method', '')
    if 'gptq_mix' in m and d.get('model') == 'Qwen2.5-0.5B':
        ratio = m.replace('gptq_mix', '')
        gptq_mix[ratio] = d['scores'].get('gsm8k', 0)
    elif 'config_b_mix' in m and d.get('model') == 'Qwen2.5-0.5B':
        ratio = m.replace('config_b_mix', '')
        config_b_mix[ratio] = d['scores'].get('gsm8k', 0)

print('GPTQ Mixture GSM8K Scores:')
for ratio, score in sorted(gptq_mix.items(), key=lambda x: -x[1]):
    wiki, gsm = ratio.split('_')
    print(f'  {ratio} (wiki={wiki}%, gsm8k={gsm}%): {score}')

top2_gptq = sorted(gptq_mix.items(), key=lambda x: -x[1])[:2]
print(f'  Top-2 GPTQ: {top2_gptq[0][0]}, {top2_gptq[1][0]}')

print()
print('Config_B Mixture GSM8K Scores:')
for ratio, score in sorted(config_b_mix.items(), key=lambda x: -x[1]):
    wiki, gsm = ratio.split('_')
    print(f'  {ratio} (wiki={wiki}%, gsm8k={gsm}%): {score}')

top2_cb = sorted(config_b_mix.items(), key=lambda x: -x[1])[:2]
print(f'  Top-2 Config_B: {top2_cb[0][0]}, {top2_cb[1][0]}')

print()
print('Phase 2 command:')
ratios = ','.join(sorted(set(r for r,_ in top2_gptq + top2_cb)))
print(f'  python experiments/exp16_calib_mixture/run.py --phase 2 --ratios {ratios}')
print(f'  (This runs {len(top2_gptq) + len(top2_cb)} combinations with full 4-task eval)')
