#!/usr/bin/env python3
"""Read-only check of the per-session manual maskinfos against the Wang ones in prfprepare.

usage: python check_session_maskinfo.py PREP_DIR SAVE_DIR [--dest] [--subjects sub-01 ...]
  PREP_DIR = .../prfprepare/analysis-02_votcROI
  SAVE_DIR = save/EHUMiguelMartinez
  --dest   = check the copies inside PREP_DIR/sub-XX/ses-YY/func instead of
             SAVE_DIR/sub-XX/{LH,RH}/maskinfo_sessions/ses-YY  (use after copying)

Per sub / ses / hemi / variant it checks:
  all 11 ROI files present, header == that session's V1v-wang maskinfo (boldFiles,
  thisHemiSize, hemisphere, key order), atlas/roi match the file name, boldFiles all
  from this sub-ses, roiIndBold == the session's fs->BOLD mapping, and an exact
  round trip against the mgz. Without --dest it also reports name collisions in PREP_DIR.
Then prints one line per sub-ses with the tasks listed and the bar width.
"""
import csv, glob, json, os, re, sys
import numpy as np
import nibabel as nib

ROI = {1: 'V1v', 2: 'V1d', 3: 'V2v', 4: 'V2d', 5: 'V3v', 6: 'V3d',
       7: 'hV4', 8: 'VO1', 9: 'V3a', 10: 'IPS0', 11: 'IPS1'}
TAG = 'miguel'
VARIANTS = ('withfovea', 'nofovea')
STIM_CSV = '/bcbl/home/public/Gari/VOTCLOC/main_exp/BIDS/sourcedata/stim_params_summary.csv'


def tasks_of(bold_files):
    return sorted({re.search(r'task-([^_]+)', b).group(1) for b in bold_files})


def bar_widths():
    try:
        rows = csv.DictReader(open(STIM_CSV))
        return {(f'sub-{r["sub"]}', f'ses-{r["ses"]}'): r['barWidth'] for r in rows}
    except OSError:
        return {}


def check(prep, save, dest, subjects=None):
    widths = bar_widths()
    n_err = n_files = 0
    summary = {}
    for sub_dir in sorted(glob.glob(os.path.join(save, 'sub-*'))):
        sid = os.path.basename(sub_dir)
        if subjects and sid not in subjects:
            continue
        for H in ('LH', 'RH'):
            hemi = H[0]
            labels = {}
            for variant in VARIANTS:
                mgz = glob.glob(os.path.join(sub_dir, H, f'{sid}_hemi-{H}_*_{variant}.mgz'))
                labels[variant] = np.asarray(nib.load(mgz[0]).dataobj).reshape(-1).astype(int)
            for ses_dir in sorted(glob.glob(os.path.join(prep, sid, 'ses-*'))):
                ses = os.path.basename(ses_dir)
                func = os.path.join(ses_dir, 'func')
                tpl_f = os.path.join(func, f'{sid}_{ses}_hemi-{hemi}_desc-V1v-wang_maskinfo.json')
                if not os.path.isfile(tpl_f):
                    continue
                tpl = json.load(open(tpl_f))
                fs2bold = {}
                for f in glob.glob(os.path.join(func, f'{sid}_{ses}_hemi-{hemi}_desc-*_maskinfo.json')):
                    if TAG in os.path.basename(f):
                        continue
                    d = json.load(open(f))
                    fs2bold.update(zip(d['roiIndFsnative'], d['roiIndBold']))
                in_union = np.zeros(labels[VARIANTS[0]].size, bool)
                in_union[list(fs2bold)] = True
                src = func if dest else os.path.join(sub_dir, H, 'maskinfo_sessions', ses)
                errs = []
                for variant in VARIANTS:
                    L = labels[variant]
                    rebuilt = np.zeros_like(L)
                    for k, roi in ROI.items():
                        fname = f'{sid}_{ses}_hemi-{hemi}_desc-{roi}-{TAG}{variant}_maskinfo.json'
                        f = os.path.join(src, fname)
                        if not os.path.isfile(f):
                            if (L == k).any():
                                errs.append(f'missing {fname}')
                            continue
                        n_files += 1
                        if not dest and os.path.exists(os.path.join(func, fname)):
                            errs.append(f'already exists in prfprepare: {fname}')
                        d = json.load(open(f))
                        if list(d) != list(tpl):
                            errs.append(f'{roi}-{variant}: key order differs from Wang')
                        for key in ('boldFiles', 'thisHemiSize', 'hemisphere'):
                            if d[key] != tpl[key]:
                                errs.append(f'{roi}-{variant}: {key} != V1v-wang')
                        if d['atlas'] != f'{TAG}{variant}' or d['roi'] != roi:
                            errs.append(f'{roi}-{variant}: atlas/roi fields')
                        if any(not b.startswith(f'{sid}_{ses}_') for b in d['boldFiles']):
                            errs.append(f'{roi}-{variant}: boldFiles from another session')
                        if d['roiIndBold'] != [fs2bold.get(v, -1) for v in d['roiIndFsnative']]:
                            errs.append(f'{roi}-{variant}: roiIndBold != session mapping')
                        rebuilt[d['roiIndFsnative']] = k
                    if not np.array_equal(rebuilt, np.where(in_union, L, 0)):
                        errs.append(f'{variant}: round trip mismatch')
                n_err += len(errs)
                for e in errs[:5]:
                    print(f'  ERROR {sid} {ses} hemi-{hemi}: {e}')
                if len(errs) > 5:
                    print(f'  ... {len(errs) - 5} more errors in {sid} {ses} hemi-{hemi}')
                summary.setdefault((sid, ses), {})[hemi] = tasks_of(tpl['boldFiles'])

    print('\nsub     ses     width  tasks listed (L | R if different)')
    for (sid, ses), t in sorted(summary.items()):
        listed = [x for x in t.get('L', t.get('R')) if x != 'fLoc']
        other = [x for x in t.get('R', []) if x != 'fLoc']
        note = '' if not other or other == listed else f'  | R: {other}'
        flag = '' if 'retFF' in listed or 'retfixFF' in listed else '   <- no FF'
        print(f'{sid}  {ses}  {widths.get((sid, ses), "?"):>5}  {listed}{note}{flag}')
    print(f'\n{n_files} files checked, {n_err} errors')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('prep'); ap.add_argument('save')
    ap.add_argument('--dest', action='store_true')
    ap.add_argument('--subjects', nargs='*')
    a = ap.parse_args()
    check(a.prep, a.save, a.dest, a.subjects)
