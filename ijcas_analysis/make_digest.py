"""Builds RESULTS_DIGEST.md = digest_header.md + raw JSON of every result file used."""
import json, glob, os
H = os.path.dirname(os.path.abspath(__file__))
out = [open(f'{H}/digest_header.md').read()]
for pat in ('deep_*.json', 'closedloop_*.json', 'weight_select_train_*.json', 'final_*.json'):
    for f in sorted(glob.glob(f'{H}/results/{pat}')):
        out.append(f'\n### {os.path.basename(f)}\n```json\n{json.dumps(json.load(open(f)), indent=1)}\n```\n')
open(f'{H}/RESULTS_DIGEST.md', 'w').write(''.join(out)); print('written', sum(len(x) for x in out), 'chars')
