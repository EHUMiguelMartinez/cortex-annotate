#!/usr/bin/env python3
"""Copy the staged per-session manual maskinfos into prfprepare (dry run by default).

usage: python copy_session_maskinfo.py PREP_DIR SAVE_DIR [--subjects sub-01 ...] [--apply] [--force]
       python copy_session_maskinfo.py PREP_DIR SAVE_DIR --undo MANIFEST [--apply]
  PREP_DIR = .../prfprepare/analysis-02_votcROI
  SAVE_DIR = save/EHUMiguelMartinez

  SAVE_DIR/sub-XX/{LH,RH}/maskinfo_sessions/ses-YY/<file>  ->  PREP_DIR/sub-XX/ses-YY/func/<file>

Only files named sub-XX_ses-YY_hemi-[LR]_desc-<ROI>-miguel(withfovea|nofovea)_maskinfo.json
are copied, and only into an existing func/ of the same sub-ses. Existing destination files
are skipped (identical or not) unless --force. Every copy is read back and compared.
With --apply a manifest (src, dest, md5) is written to SAVE_DIR/copy_manifest_<time>.tsv;
--undo MANIFEST deletes exactly those files, and only while their md5 is unchanged.
"""
import argparse, glob, hashlib, os, re, shutil, sys, time

ROIS = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'VO1', 'V3a', 'IPS0', 'IPS1']
NAME = re.compile(r'(sub-\d+)_(ses-\d+)_hemi-([LR])_desc-(%s)-miguel(withfovea|nofovea)_maskinfo\.json$'
                  % '|'.join(ROIS))


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def plan(prep, save, subjects):
    """Yield (src, dest, problem-or-None) for every staged file."""
    for src in sorted(glob.glob(os.path.join(save, 'sub-*', '[LR]H', 'maskinfo_sessions', 'ses-*', '*'))):
        parts = src.split(os.sep)
        sid, H, ses, name = parts[-5], parts[-4], parts[-2], parts[-1]
        if subjects and sid not in subjects:
            continue
        m = NAME.match(name)
        func = os.path.join(prep, sid, ses, 'func')
        dest = os.path.join(func, name)
        if not m:
            yield src, dest, 'name not allowed'
        elif (m.group(1), m.group(2), m.group(3)) != (sid, ses, H[0]):
            yield src, dest, 'sub/ses/hemi in name != folder'
        elif not os.path.isdir(func):
            yield src, dest, 'destination func/ does not exist'
        else:
            yield src, dest, None


def copy(prep, save, subjects, apply, force):
    counts = {'copy': 0, 'overwrite': 0, 'identical': 0, 'differs': 0, 'refused': 0, 'failed': 0}
    manifest = []
    for src, dest, problem in plan(prep, save, subjects):
        if problem:
            counts['refused'] += 1
            print(f'REFUSED  {src}: {problem}'); continue
        if os.path.exists(dest):
            if md5(dest) == md5(src):
                counts['identical'] += 1; continue
            if not force:
                counts['differs'] += 1
                print(f'EXISTS, DIFFERS (skipped, use --force): {dest}'); continue
            action = 'overwrite'
        else:
            action = 'copy'
        if apply:
            shutil.copyfile(src, dest)
            if md5(dest) != md5(src):
                counts['failed'] += 1
                print(f'FAILED READ-BACK  {dest}'); continue
            manifest.append((src, dest, md5(dest)))
        counts[action] += 1

    per_sub = {}
    for src, dest, _ in manifest:
        sid = os.path.basename(dest).split('_')[0]
        per_sub[sid] = per_sub.get(sid, 0) + 1
    print(('APPLIED' if apply else 'DRY RUN (nothing written; add --apply)') + ': ' +
          ', '.join(f'{k}={v}' for k, v in counts.items()))
    if apply and manifest:
        out = os.path.join(save, f'copy_manifest_{time.strftime("%Y%m%d-%H%M%S")}.tsv')
        with open(out, 'w') as fh:
            fh.write('src\tdest\tmd5\n')
            fh.writelines(f'{s}\t{d}\t{h}\n' for s, d, h in manifest)
        print(f'manifest: {out}  ({len(manifest)} files; {per_sub})')


def undo(manifest, apply):
    rows = [l.rstrip('\n').split('\t') for l in open(manifest)][1:]
    n_del = n_keep = 0
    for _, dest, h in rows:
        if not os.path.exists(dest):
            continue
        if not NAME.match(os.path.basename(dest)) or md5(dest) != h:
            n_keep += 1
            print(f'KEPT (changed or unexpected name): {dest}'); continue
        if apply:
            os.remove(dest)
        n_del += 1
    print(('UNDONE' if apply else 'UNDO DRY RUN (add --apply)') +
          f': {n_del} files {"deleted" if apply else "would be deleted"}, {n_keep} kept')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('prep'); ap.add_argument('save')
    ap.add_argument('--subjects', nargs='*')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--undo')
    a = ap.parse_args()
    if a.undo:
        undo(a.undo, a.apply)
    else:
        copy(a.prep, a.save, a.subjects, a.apply, a.force)
