#!/usr/bin/env python3
"""Convert cortex-annotate label .mgz files into one maskinfo JSON per ROI, per session (for each sub-hemi).

usage: python mgz_to_maskinfo.py PREP_DIR MGZ [MGZ ...]
  PREP_DIR = .../prfprepare/analysis-02_votcROI
  MGZ      = sub-XX_hemi-LH_<roi tag>_withfovea.mgz / _nofovea.mgz
Output goes to  <folder of the mgz>/maskinfo_sessions/ses-YY/  (staging, final pipeline names).
Each session copies its header (boldFiles, thisHemiSize) from its own V1v-wang maskinfo
and its fs -> BOLD mapping from its own maskinfos.

Run:
python3 scripts/mgz_to_maskinfo.py \
  /bcbl/home/public/Gari/VOTCLOC/main_exp/BIDS/derivatives/prfprepare/analysis-02_votcROI \
  save/EHUMiguelMartinez/sub-*/*/*.mgz
"""
import glob, json, os, re, sys
import numpy as np

try:
    import nibabel as nib
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: nibabel is not installed in the active Python environment.\n"
        "Install it with one of:\n"
        "  conda install -c conda-forge nibabel\n"
        "  pip install nibabel\n"
        "Then rerun the script."
    ) from exc
 
ROI = {1: 'V1v', 2: 'V1d', 3: 'V2v', 4: 'V2d', 5: 'V3v', 6: 'V3d',
       7: 'hV4', 8: 'VO1', 9: 'V3a', 10: 'IPS0', 11: 'IPS1'}
TAG = 'miguel'          # -> desc-<ROI>-miguelnofovea / -miguelwithfovea
TEMPLATE_ROI = 'V1v-wang'   # header source; NOT the first file alphabetically (LH *-words-wang carry ses-01 boldFiles)


def session_maps(prep, sid, ses, hemi):
    """fs vertex -> BOLD position for one session, plus that session's V1v-wang maskinfo as template."""
    func = os.path.join(prep, sid, ses, 'func')
    files = sorted(f for f in glob.glob(os.path.join(
        func, f'{sid}_{ses}_hemi-{hemi}_desc-*_maskinfo.json'))
        if TAG not in os.path.basename(f))
    if not files:
        return None, None
    fs2bold = {}
    for f in files:
        d = json.load(open(f))
        for fs, pos in zip(d['roiIndFsnative'], d['roiIndBold']):
            fs2bold[int(fs)] = int(pos)
    tpl = os.path.join(func, f'{sid}_{ses}_hemi-{hemi}_desc-{TEMPLATE_ROI}_maskinfo.json')
    template = json.load(open(tpl)) if os.path.isfile(tpl) else None
    return fs2bold, template


def tasks_of(bold_files):
    return sorted({re.search(r'task-([^_]+)', b).group(1) for b in bold_files})


def convert(prep, mgz):
    name = os.path.basename(mgz)
    m = re.match(r'(sub-\d+)_hemi-([LR])H?_.*_(withfovea|nofovea)\.mgz$', name)
    if not m:
        print('skip (name not recognised):', name); return
    sid, hemi, variant = m.groups()
    labels = np.asarray(nib.load(mgz).dataobj).reshape(-1).astype(int)
    out_root = os.path.join(os.path.dirname(os.path.abspath(mgz)), 'maskinfo_sessions')
    print(f'{sid} hemi-{hemi} {variant}')

    n_ses = 0
    for ses_dir in sorted(glob.glob(os.path.join(prep, sid, 'ses-*'))):
        ses = os.path.basename(ses_dir)
        fs2bold, template = session_maps(prep, sid, ses, hemi)
        if fs2bold is None:
            print(f'  {ses}: no maskinfo found -> skipped'); continue
        if template is None:
            print(f'  {ses}: no {TEMPLATE_ROI} maskinfo -> skipped'); continue
        bad = [b for b in template['boldFiles'] if not b.startswith(f'{sid}_{ses}_')]
        if bad:
            print(f'  {ses}: WARNING template boldFiles from another session: {bad[:3]}')
        if len(fs2bold) != template['thisHemiSize']:
            print(f'  {ses}: NOTE union of existing maskinfos covers {len(fs2bold)} of '
                  f'{template["thisHemiSize"]} BOLD positions')

        out_dir = os.path.join(out_root, ses)
        os.makedirs(out_dir, exist_ok=True)
        counts = []
        for k, roi in ROI.items():
            fs_ids = np.flatnonzero(labels == k)
            if fs_ids.size == 0:
                counts.append(f'{roi}:EMPTY'); continue
            keep = [int(v) for v in fs_ids if int(v) in fs2bold]
            d = dict(template)
            d['atlas'] = f'{TAG}{variant}'
            d['roi'] = roi
            d['roiIndFsnative'] = keep
            d['roiIndBold'] = [fs2bold[v] for v in keep]
            out = os.path.join(out_dir,
                  f'{sid}_{ses}_hemi-{hemi}_desc-{roi}-{TAG}{variant}_maskinfo.json')
            with open(out, 'w') as fh:
                json.dump(d, fh, indent=4)
            counts.append(f'{roi}:{len(keep)}/{fs_ids.size}' if len(keep) != fs_ids.size
                          else f'{roi}:{len(keep)}')

        # round trip: rebuild labels from the JSONs and compare
        rebuilt = np.zeros_like(labels)
        for k, roi in ROI.items():
            f = os.path.join(out_dir, f'{sid}_{ses}_hemi-{hemi}_desc-{roi}-{TAG}{variant}_maskinfo.json')
            if os.path.isfile(f):
                rebuilt[json.load(open(f))['roiIndFsnative']] = k
        expected = np.where(np.isin(np.arange(labels.size), list(fs2bold)), labels, 0)
        ok = 'round trip OK' if np.array_equal(rebuilt, expected) else 'ROUND TRIP MISMATCH'
        print(f'  {ses}: tasks={tasks_of(template["boldFiles"])} {ok}')
        print(f'         {" ".join(counts)}')
        n_ses += 1
    if n_ses == 0:
        print(f'  no session with Wang maskinfo found')
 
 
if __name__ == '__main__':
    for p in sys.argv[2:]:
        convert(sys.argv[1], p)