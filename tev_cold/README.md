# Cold-adapted TEV protease — working notes

Goal: a TEV protease variant with higher catalytic efficiency at 4 °C, for
fusion-tag cleavage at low temperature.

Plan: **rational library + bench screen now (Phase 1), EVB/MD activation-parameter
calculations built in parallel (Phase 2), retrospective validation of the
calculations against our own measured variants (Phase 3).** Phase 3 is the
publishable claim; Phases 1 and 2 are each useful on their own.

## Why not AlphaProtein Novo

This directory lives in an AP Novo fork, so the honest scoping first. AP Novo is
a *de novo* motif-scaffolding pipeline, and it is the wrong primary tool here:

- **No temperature objective exists anywhere in the model or the metrics.** The
  only `b_factor` use in `src/alphaprotein_novo/metrics/` is storing plDDT
  (`confidence_metrics.py:107`) and masking motif residues
  (`kemp_metrics.py:410`). There is no ΔH‡, flexibility, or RMSF term. The
  pipeline cannot rank designs by the property we care about.
- **The substrate cannot be represented.** `motif_spec.py:399` enforces
  `'Expected source ligand structure to have only 1 ligand'` — each ligand chain
  must be a single chemical component. TEV's substrate is a ≥7-residue peptide.
- **No proteolysis evaluation suite.** `evaluate_design.py:391-395` provides
  kemp_eliminase, serine_esterase, dehp_esterase, carbene_transfer,
  nitrene_transfer. All small-molecule chemistry.
- **Rebuilding the scaffold would destroy the specificity we need.** TEV's
  ENLYFQ↓S/G selectivity comes from an extended S1–S6 groove formed by several
  loops; a fresh scaffold holding only the triad is not a TEV-specificity
  protease.

AP Novo's partial-diffusion mode (`examples/dehp_esterase_partial_diffusion`,
600 steps → TM ≈ 95 to parent) *can* diversify surface loops while holding the
catalytic core, and it is a reasonable generator of loop variants later. It still
contributes no cold-adaptation signal, so it is not where we start.

## Design hypothesis

Cold adaptation is mechanistically specific, which is what makes it designable.
Literature retrieved and verified via PubMed:

- Psychrophilic enzymes reach higher low-temperature k_cat through a **lower
  activation enthalpy, offset by a more negative activation entropy**. Lonhienne,
  Gerday & Feller argue the entropic penalty is contained by **keeping some
  domains rigid while softening only the structures catalysis requires** —
  [10.1016/s0167-4838(00)00210-7](https://doi.org/10.1016/s0167-4838(00)00210-7)
  (PMID 11087936).
- Koenekoop & Åqvist find this ΔH‡/ΔS‡ redistribution "appears to be a universal
  feature of psychrophilic enzymes", originating "from a higher flexibility of
  certain parts of the **protein surface**" —
  [10.1093/molbev/msad099](https://doi.org/10.1093/molbev/msad099) (PMID 37116207).
- **Closest precedent.** In psychrophilic elastase — a chymotrypsin-fold
  protease, the same fold class as TEV — Sočan et al. link cold adaptation to
  surface-exposed loop mobility and report that "calculations of mutations in a
  single surface loop show that the temperature dependence of the catalytic
  reaction is altered in a predictable way" —
  [10.1021/acs.biochem.8b00078](https://doi.org/10.1021/acs.biochem.8b00078)
  (PMID 29726678). This is the paper to reproduce in Phase 2.
- Assay design caveat: Arrhenius plots curve for reasons other than unfolding, so
  measure a real temperature series, not two points. Arcus & Mulholland,
  [10.1146/annurev-biophys-121219-081520](https://doi.org/10.1146/annurev-biophys-121219-081520);
  Åqvist, Sočan & Purg on hidden conformational states and temperature optima,
  [10.1021/acs.biochem.0c00705](https://doi.org/10.1021/acs.biochem.0c00705).

So: **soften solvent-exposed, coil-like segments distal from the catalytic triad;
keep the core and the substrate groove rigid.**

### TEV-specific constraints

- **Build every variant on the S219V background.** Kapust et al. report S219V is
  ~100-fold more resistant to autolysis *and* a more efficient catalyst —
  [10.1093/protein/14.12.993](https://doi.org/10.1093/protein/14.12.993)
  (PMID 11809930). Loosening surface loops on wild-type TEV will accelerate
  self-cleavage and confound the kinetics.
- **Structural template.** Phan et al. solved the inactive C151A mutant as a
  substrate complex (2.2 Å) and autolysis-resistant S219D as a product complex
  (1.8 Å) — [10.1074/jbc.M207224200](https://doi.org/10.1074/jbc.M207224200)
  (PMID 12377789). Use the substrate complex so the peptide occludes the groove
  during the SASA calculation.
- **Do not touch the specificity subsites.** Tözsér et al. map S3/S4 as the main
  determinants distinguishing TEV from TVMV protease —
  [10.1111/j.1742-4658.2004.04493.x](https://doi.org/10.1111/j.1742-4658.2004.04493.x)
  (PMID 15654889).

Expect a thermostability cost. That is the documented trade-off, and it is
acceptable for a reagent used at 4 °C — but it will shorten shelf life, so
include a storage-stability check.

## Phase 1 — rational library

1. Fetch the Phan et al. substrate complex. **Confirm the PDB accession from the
   paper itself rather than trusting a remembered ID**, and confirm whether the
   deposited numbering is author or internal before using residue numbers
   anywhere (AP Novo manifests make the same distinction via `is_author_naming`).
2. Confirm the catalytic triad residue numbers in that specific entry.
3. Rank candidate positions:

   ```bash
   python tev_cold/select_flexibility_positions.py <structure>.cif \
       --protein_chain A \
       --catalytic <triad numbers> \
       --substrate_chain <peptide chain> \
       --out tev_positions.tsv
   ```

   The script masks the triad, the active-site second shell (8 Å), the
   substrate-contacting subsites (5 Å), buried residues (<40 Å² SASA), and
   regular secondary structure; it then ranks what survives by exposure and
   distance from the triad, and reports contiguous runs. **Prefer contiguous runs
   over isolated positions** — the precedent is a whole surface loop, not a point
   mutation. Install `mkdssp` for real secondary structure; the built-in
   CA-geometry fallback is triage only.
4. Cross-reference against potyviral NIa protease homologs. Positions already
   variable across the family are safer to mutate; a conserved surface residue
   probably has a job. Sočan et al. found cold/mesophilic mobility differences
   tracked family conservation patterns.
5. Pick 20–50 variants: Gly/Ser substitutions and surface salt-bridge removals
   in the top-ranked runs, as single mutants plus a few per-loop combinations.
6. Fold-check with AF3 (or AF3-LA via this repo) that the triad geometry survives
   — a cheap negative filter, *not* an activity prediction.
7. Screen on the bench: temperature series from 4 to 30 °C, ENLYFQ↓G substrate,
   measure k_cat/K_M at each point, then decompose into ΔH‡ and ΔS‡.

## Phase 2 — EVB/MD activation parameters

Reproduce the Sočan/Åqvist protocol on TEV: empirical-valence-bond MD of the
acylation step across 5–6 temperatures, replicated, with Arrhenius decomposition
into ΔH‡ and ΔS‡. Run on the A800 cluster.

Be realistic about scope: EVB parameterization for a **cysteine** protease
nucleophile is a substantial piece of work in its own right, and the published
precedent is a serine protease. Budget for that, and consider validating the
pipeline first by reproducing the published elastase result before trusting it
on TEV.

## Phase 3 — validation

Test whether computed ΔH‡ predicts the measured low-temperature k_cat/K_M across
our own variant panel. A positive result makes the computational protocol
transferable; a negative result is still worth reporting, given how widely the
flexibility–activity narrative is assumed.

## Environment

```bash
bash tev_cold/setup_env.sh          # venv, weights, build_data, GPU smoke test
```

`alphafold3` is pinned to a specific commit in `pyproject.toml` (upstream floats
on HEAD, which would silently change the folding stage mid-campaign). Bump it
deliberately, not accidentally.

## Licensing

The AP Novo generator weights are **non-commercial use only** and must not be
shared outside the organization; outputs must not be used to train or distil
other protein-design models. See `WEIGHTS_TERMS_OF_USE.md`. As an academic lab we
are in scope, but the output-reuse restriction matters if we later want to train
anything on generated structures.

Literature in this document was retrieved from PubMed; DOIs link to the
originals.
