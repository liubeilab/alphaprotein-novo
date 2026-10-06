#!/usr/bin/env python3
"""Rank candidate positions for cold-adaptation loop softening.

Design hypothesis implemented here (see tev_cold/README.md for citations):
cold-adapted enzymes gain low-temperature k_cat through a lower activation
enthalpy, and that effect has been traced to higher mobility of *surface*
regions rather than of the active site itself. The catalytic machinery and the
substrate-binding subsites should therefore stay rigid, while mutations that
add backbone flexibility are placed on solvent-exposed, non-contacting,
coil-like segments that are distal from the catalytic residues.

This script does the geometric bookkeeping for that selection. It does NOT
predict activity, activation enthalpy, or any thermodynamic quantity -- it
narrows several hundred positions down to a list short enough to reason about
by hand or to feed to an MD/EVB calculation.

Example:
  python select_flexibility_positions.py structure.cif \
      --protein_chain A --catalytic A46,A81,A151 --substrate_chain B

Secondary structure: uses mkdssp when it is on PATH, otherwise falls back to a
CA-geometry heuristic that is only good enough for triage. Prefer real DSSP
before committing to a library.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley

# Solvent accessibility above which a residue is treated as surface-exposed.
# Absolute Shrake-Rupley SASA in square angstroms, deliberately a plain
# threshold rather than a normalised value so there is no hidden reference
# dataset. Tune with --sasa_min.
DEFAULT_SASA_MIN = 40.0

# Residues closer than this to any substrate/ligand atom are treated as
# binding-subsite residues and excluded: they carry the specificity.
DEFAULT_SUBSTRATE_CUTOFF = 5.0

# Residues closer than this to any catalytic atom are excluded as active-site
# second shell.
DEFAULT_CATALYTIC_CUTOFF = 8.0


@dataclass
class Residue:
  chain: str
  number: int
  name: str
  sasa: float = 0.0
  min_dist_catalytic: float = float('inf')
  min_dist_substrate: float = float('inf')
  ss: str = '-'
  ca: np.ndarray | None = None
  reasons: list[str] = field(default_factory=list)

  @property
  def label(self) -> str:
    return f'{self.name}{self.number}'

  @property
  def is_candidate(self) -> bool:
    return not self.reasons


def _parse_structure(path: str):
  parser = MMCIFParser(QUIET=True) if path.endswith('.cif') else PDBParser(
      QUIET=True
  )
  return parser.get_structure('input', path)[0]


def _parse_catalytic(spec: str) -> set[tuple[str, int]]:
  """Parses 'A46,A81,A151' or '46,81,151' into (chain, resnum) pairs."""
  out = set()
  for token in spec.split(','):
    token = token.strip()
    if not token:
      continue
    if token[0].isalpha():
      out.add((token[0], int(token[1:])))
    else:
      out.add(('', int(token)))
  return out


def _dssp_secondary_structure(model, protein_chain: str) -> dict[int, str]:
  """Runs mkdssp if available; returns {resnum: ss_code}, empty on failure."""
  if not shutil.which('mkdssp'):
    return {}
  try:
    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
      io = PDBIO()
      io.set_structure(model)
      io.save(tmp.name)
      pdb_path = tmp.name
    from Bio.PDB.DSSP import DSSP  # local import: optional dependency path

    dssp = DSSP(model, pdb_path, dssp='mkdssp')
    return {
        key[1][1]: dssp[key][2]
        for key in dssp.keys()
        if key[0] == protein_chain
    }
  except Exception as exc:  # noqa: BLE001 - triage tool, never fatal
    print(f'  [warn] DSSP failed ({exc}); using CA-geometry heuristic',
          file=sys.stderr)
    return {}


def _heuristic_secondary_structure(residues: list[Residue]) -> None:
  """Crude coil/helix/strand call from CA(i)->CA(i+3) and CA(i)->CA(i+4).

  In a regular alpha-helix CA(i)-CA(i+3) is about 5.0-5.5 A and CA(i)-CA(i+4)
  about 6.2 A; in an extended strand CA(i)-CA(i+4) exceeds 12 A. Helix is only
  called when BOTH separations are in range, because the i->i+4 test alone also
  matches compact irregular turns and badly over-calls helix. Anything else, or
  any position where the window is broken by a chain gap, is reported as coil.
  This is triage only -- use mkdssp before committing to a library.
  """
  by_number = {r.number: r for r in residues}
  for r in residues:
    nxt3 = by_number.get(r.number + 3)
    nxt4 = by_number.get(r.number + 4)
    if (r.ca is None or nxt3 is None or nxt4 is None
        or nxt3.ca is None or nxt4.ca is None):
      r.ss = 'C'
      continue
    d3 = float(np.linalg.norm(r.ca - nxt3.ca))
    d4 = float(np.linalg.norm(r.ca - nxt4.ca))
    if 4.5 <= d3 <= 6.3 and 5.0 <= d4 <= 7.3:
      r.ss = 'H'
    elif d4 > 11.5:
      r.ss = 'E'
    else:
      r.ss = 'C'


def analyse(
    path: str,
    protein_chain: str,
    catalytic_spec: str,
    substrate_chains: list[str],
    sasa_min: float,
    substrate_cutoff: float,
    catalytic_cutoff: float,
) -> list[Residue]:
  model = _parse_structure(path)

  # SASA is computed on the complete model so that bound substrate correctly
  # occludes the groove; a protein-only calculation would mark subsite
  # residues as exposed.
  ShrakeRupley().compute(model, level='R')

  if protein_chain not in {c.id for c in model}:
    raise SystemExit(
        f'chain {protein_chain!r} not in structure '
        f'(found {sorted(c.id for c in model)})'
    )

  catalytic = _parse_catalytic(catalytic_spec)
  catalytic_numbers = {num for _, num in catalytic}

  residues: list[Residue] = []
  for res in model[protein_chain]:
    if not is_aa(res, standard=False):
      continue
    ca = res['CA'].coord if 'CA' in res else None
    residues.append(
        Residue(
            chain=protein_chain,
            number=res.id[1],
            name=res.get_resname(),
            sasa=float(getattr(res, 'sasa', 0.0)),
            ca=None if ca is None else np.asarray(ca, dtype=float),
        )
    )

  if not residues:
    raise SystemExit(f'no amino-acid residues found in chain {protein_chain}')

  # Catalytic atom cloud.
  cat_atoms = [
      a.coord
      for res in model[protein_chain]
      if res.id[1] in catalytic_numbers
      for a in res
  ]
  if not cat_atoms:
    raise SystemExit(
        f'none of the catalytic residues {sorted(catalytic_numbers)} were '
        f'found in chain {protein_chain}; check --catalytic numbering '
        '(author vs internal)'
    )
  cat_atoms = np.asarray(cat_atoms, dtype=float)

  # Substrate / ligand atom cloud.
  sub_atoms = []
  for chain_id in substrate_chains:
    if chain_id not in {c.id for c in model}:
      print(f'  [warn] substrate chain {chain_id!r} absent; subsite masking '
            'will be incomplete', file=sys.stderr)
      continue
    sub_atoms.extend(a.coord for res in model[chain_id] for a in res)
  sub_atoms = np.asarray(sub_atoms, dtype=float) if sub_atoms else None

  for r in residues:
    res_obj = model[protein_chain][r.number]
    coords = np.asarray([a.coord for a in res_obj], dtype=float)
    r.min_dist_catalytic = float(
        np.min(np.linalg.norm(coords[:, None, :] - cat_atoms[None, :, :],
                              axis=-1))
    )
    if sub_atoms is not None:
      r.min_dist_substrate = float(
          np.min(np.linalg.norm(coords[:, None, :] - sub_atoms[None, :, :],
                                axis=-1))
      )

  ss_map = _dssp_secondary_structure(model, protein_chain)
  if ss_map:
    for r in residues:
      r.ss = ss_map.get(r.number, '-')
  else:
    _heuristic_secondary_structure(residues)

  # Exclusion logic -- every rejected position records why.
  for r in residues:
    if r.number in catalytic_numbers:
      r.reasons.append('catalytic')
    if r.min_dist_catalytic < catalytic_cutoff:
      r.reasons.append(f'active-site<{catalytic_cutoff:g}A')
    if r.min_dist_substrate < substrate_cutoff:
      r.reasons.append(f'subsite<{substrate_cutoff:g}A')
    if r.sasa < sasa_min:
      r.reasons.append(f'buried<{sasa_min:g}A2')
    if r.ss in ('H', 'G', 'I', 'E', 'B'):
      r.reasons.append(f'structured:{r.ss}')
    if r.name in ('GLY', 'PRO'):
      # Gly is already maximally flexible; Pro removal is a separate, more
      # aggressive strategy that deserves its own hypothesis.
      r.reasons.append(f'skip:{r.name}')

  return residues


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('structure', help='input .cif or .pdb')
  ap.add_argument('--protein_chain', default='A')
  ap.add_argument('--catalytic', required=True,
                  help='comma-separated catalytic residue numbers, e.g. 46,81,151')
  ap.add_argument('--substrate_chain', default='',
                  help='comma-separated chain ids of substrate/ligand, if present')
  ap.add_argument('--sasa_min', type=float, default=DEFAULT_SASA_MIN)
  ap.add_argument('--substrate_cutoff', type=float,
                  default=DEFAULT_SUBSTRATE_CUTOFF)
  ap.add_argument('--catalytic_cutoff', type=float,
                  default=DEFAULT_CATALYTIC_CUTOFF)
  ap.add_argument('--top', type=int, default=40,
                  help='how many ranked candidates to print')
  ap.add_argument('--out', default='', help='write full TSV table here')
  args = ap.parse_args()

  substrate_chains = [c for c in args.substrate_chain.split(',') if c]
  residues = analyse(
      args.structure,
      args.protein_chain,
      args.catalytic,
      substrate_chains,
      args.sasa_min,
      args.substrate_cutoff,
      args.catalytic_cutoff,
  )

  candidates = [r for r in residues if r.is_candidate]
  # Rank: most exposed and most distal first. Both terms are scale-free so the
  # ranking is not dominated by either.
  max_sasa = max((r.sasa for r in candidates), default=1.0) or 1.0
  max_dist = max((r.min_dist_catalytic for r in candidates), default=1.0) or 1.0
  candidates.sort(
      key=lambda r: -(r.sasa / max_sasa + r.min_dist_catalytic / max_dist)
  )

  print(f'structure           : {args.structure}')
  print(f'protein chain       : {args.protein_chain}  ({len(residues)} aa)')
  print(f'catalytic residues  : {args.catalytic}')
  print(f'substrate chains    : {substrate_chains or "none given"}')
  print(f'candidate positions : {len(candidates)} / {len(residues)}')
  print()
  print('Contiguous candidate runs (loop-like stretches are the better bet '
        'than isolated positions):')
  numbers = sorted(r.number for r in candidates)
  runs, run = [], []
  for n in numbers:
    if run and n == run[-1] + 1:
      run.append(n)
    else:
      if run:
        runs.append(run)
      run = [n]
  if run:
    runs.append(run)
  for rr in sorted(runs, key=len, reverse=True):
    if len(rr) >= 2:
      print(f'  {rr[0]:>4}-{rr[-1]:<4} ({len(rr)} residues)')
  print()
  print(f'Top {min(args.top, len(candidates))} ranked candidates:')
  print(f'  {"res":>8}  {"SASA":>7}  {"d_cat":>7}  {"d_sub":>7}  ss')
  for r in candidates[: args.top]:
    d_sub = '-' if r.min_dist_substrate == float('inf') else f'{r.min_dist_substrate:7.1f}'
    print(f'  {r.label:>8}  {r.sasa:7.1f}  {r.min_dist_catalytic:7.1f}  '
          f'{d_sub}  {r.ss}')

  if args.out:
    with open(args.out, 'w', newline='') as fh:
      w = csv.writer(fh, delimiter='\t')
      w.writerow(['chain', 'resnum', 'resname', 'sasa', 'min_dist_catalytic',
                  'min_dist_substrate', 'ss', 'candidate', 'exclusions'])
      for r in sorted(residues, key=lambda x: x.number):
        w.writerow([
            r.chain, r.number, r.name, f'{r.sasa:.2f}',
            f'{r.min_dist_catalytic:.2f}',
            '' if r.min_dist_substrate == float('inf') else f'{r.min_dist_substrate:.2f}',
            r.ss, int(r.is_candidate), ';'.join(r.reasons),
        ])
    print(f'\nfull table -> {args.out}')


if __name__ == '__main__':
  main()
