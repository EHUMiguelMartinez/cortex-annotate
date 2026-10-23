#!/usr/bin/env python3
"""Convert cortex-annotate label .mgz files into one maskinfo JSON per ROI.
 
usage: python mgz_to_maskinfo.py PREP_DIR MGZ [MGZ ...]
  PREP_DIR = .../prfprepare/analysis-02_votcROI
  MGZ      = sub-XX_hemi-LH_<roi tag>_withfovea.mgz / _nofovea.mgz
Output goes to  <folder of the mgz>/maskinfo/  (staging, no session in the name).
"""
import glob, json, os, re, sys
import numpy as np
import nibabel as nib
 
ROI = {1: 'V1v', 2: 'V1d', 3: 'V2v', 4: 'V2d', 5: 'V3v', 6: 'V3d',
       7: 'hV4', 8: 'VO1', 9: 'V3a', 10: 'IPS0', 11: 'IPS1'}
TAG = 'miguel'          # -> desc-<ROI>-miguelnofovea / -miguelwithfovea
 
 
def session_maps(prep, sid, ses, hemi):
    """fs vertex -> BOLD position for one session, plus a template dict."""
    files = sorted(f for f in glob.glob(os.path.join(
        prep, sid, ses, 'func', f'{sid}_{ses}_hemi-{hemi}_desc-*_maskinfo.json'))
        if TAG not in os.path.basename(f))
    if not files:
        return None, None
    fs2bold, template = {}, None
    for f in files:
        d = json.load(open(f))
        template = template or d
        for fs, pos in zip(d['roiIndFsnative'], d['roiIndBold']):
            fs2bold[int(fs)] = int(pos)
    return fs2bold, template
 
 
def convert(prep, mgz):
    name = os.path.basename(mgz)
    m = re.match(r'(sub-\d+)_hemi-([LR])H?_.*_(withfovea|nofovea)\.mgz$', name)
    if not m:
        print('skip (name not recognised):', name); return
    sid, hemi, variant = m.groups()
    labels = np.asarray(nib.load(mgz).dataobj).reshape(-1).astype(int)
 
    # reference session + consistency check across sessions
    ref = None
    for ses_dir in sorted(glob.glob(os.path.join(prep, sid, 'ses-*'))):
        ses = os.path.basename(ses_dir)
        fs2bold, template = session_maps(prep, sid, ses, hemi)
        if fs2bold is None:
            continue
        if ref is None:
            ref = (ses, fs2bold, template)
            print(f'{sid} hemi-{hemi}: reference session {ses}; '
                  f'{len(fs2bold)} vertices in mask union, thisHemiSize={template["thisHemiSize"]}')
        elif fs2bold != ref[1]:
            print(f'  WARNING: {ses} has a DIFFERENT fs->BOLD mapping than {ref[0]} '
                  f'-> per-session conversion needed')
    if ref is None:
        print(f'{sid} hemi-{hemi}: no Wang maskinfo found'); return
    _, fs2bold, template = ref
    if len(fs2bold) != template['thisHemiSize']:
        print(f'  NOTE: union of existing maskinfos covers {len(fs2bold)} of '
              f'{template["thisHemiSize"]} BOLD positions')
 
    out_dir = os.path.join(os.path.dirname(os.path.abspath(mgz)), 'maskinfo')
    os.makedirs(out_dir, exist_ok=True)
    for k, roi in ROI.items():
        fs_ids = np.flatnonzero(labels == k)
        if fs_ids.size == 0:
            print(f'  {roi:5s} {variant:9s} EMPTY (not drawn) -> no file'); continue
        keep = [int(v) for v in fs_ids if int(v) in fs2bold]
        d = dict(template)
        d['atlas'] = TAG
        d['roi'] = roi
        d['roiIndFsnative'] = keep
        d['roiIndBold'] = [fs2bold[v] for v in keep]
        out = os.path.join(out_dir,
              f'{sid}_hemi-{hemi}_desc-{roi}-{TAG}{variant}_maskinfo.json')
        with open(out, 'w') as fh:
            json.dump(d, fh, indent=4)
        print(f'  {roi:5s} {variant:9s} {len(keep):6d} kept / {fs_ids.size:6d} drawn')
 
    # round trip: rebuild labels from the JSONs and compare
    rebuilt = np.zeros_like(labels)
    for k, roi in ROI.items():
        f = os.path.join(out_dir, f'{sid}_hemi-{hemi}_desc-{roi}-{TAG}{variant}_maskinfo.json')
        if os.path.isfile(f):
            rebuilt[json.load(open(f))['roiIndFsnative']] = k
    expected = np.where(np.isin(np.arange(labels.size), list(fs2bold)), labels, 0)
    print('  round trip OK' if np.array_equal(rebuilt, expected)
          else '  ROUND TRIP MISMATCH')
 
 
if __name__ == '__main__':
    for p in sys.argv[2:]:
        convert(sys.argv[1], p)