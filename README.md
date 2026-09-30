# The Enhanced Sampling Explorer

## [**Click here to visit the Explorer!**](https://drorlab.github.io/enhanced-sampling-explorer/dashboard)

**System.** Ace-(Ala)<sub>10</sub>-Nme in vacuum. AMBER ff14SB, 300 K, Langevin dynamics,
2 fs steps, 112 atoms.

**Software.** OpenMM 8.6.1 (CPU platform), openmmtools 0.26.0, pymbar 4.2.0.

**Why vacuum.** It is very fast to simulate, which makes it a good demo of enhanced
sampling. Its specific behavior differs from the solvated peptide, where water competes for
the backbone H-bonds and the helix is much less stable.

**Coordinate.** d<sub>ee</sub>, the distance from the ACE carbonyl C to the NME N:

| state | d<sub>ee</sub> | notes |
|---|---|---|
| α-helix | ≈ 15 Å | |
| fully extended | ≈ 33 Å | ≈ 22 kcal/mol uphill |
| compact | < 8 Å | folded over; a basin in this force field |

**Reference.** 42-window umbrella sampling + MBAR.

## View it

The dashboard is a static site in `dashboard/`, and the data is included.

```bash
python3 -m http.server 8000 --directory dashboard
# open http://localhost:8000
```

Opening `index.html` directly as a `file://` URL does not work, because browsers block
`fetch` of local files. The site also works on GitHub Pages: serve the repository root, then
open `/dashboard/`.

## Pages

- **The system & d<sub>ee</sub>:** the three representative structures, the reference free-energy profile, and what is known.
- **Comparison: all methods:**
  - a time-lapse of every method's profile as its sampling budget grows;
  - error vs cost;
  - coverage vs cost;
  - ΔF(15→30 Å) vs cost.
- **Estimators:** MBAR, WHAM (0.5 Å and 2 Å bins), umbrella integration and histogram stitching, all on the same umbrella-sampling windows.
- **One page per method.** Each has:
  - a trajectory movie, with a synced d<sub>ee</sub> / H-bond time series;
  - a live "free-energy estimate so far" plot (plus the bias for metadynamics, and the walker's temperature or umbrella over time for SAMS);
  - a lineage graph of the simulations;
  - method-specific diagnostics;
  - for T-REMD, steered MD and umbrella sampling, a grid view of all copies at once (T-REMD highlights exchanges).

## Methods

| method | sampling | free energy from |
|---|---|---|
| Plain MD | 20 ns unbiased | histogram |
| Well-tempered metadynamics | 20 ns, bias on d<sub>ee</sub>, γ = 10 | the bias |
| Gaussian accelerated MD | 2 ns cMD + 4 ns equil + 14 ns dual boost | cumulant reweighting |
| Temperature REMD | 10 replicas × 2.5 ns, 300–700 K | MBAR at 300 K |
| SAMS over temperature | 12 ns, 12 temperatures, initial weights from 12 × 120 ps runs | MBAR at 300 K |
| SAMS over temperature, uninitialized (cautionary) | 12 ns, all weights start at 0: never reaches 300 K | MBAR extrapolated to 300 K |
| SAMS over umbrellas | 15 ns, 42 umbrellas | MBAR |
| Weighted ensemble (d<sub>ee</sub> bins) | ~760 ns in 10 ps segments | bin weights |
| Weighted ensemble (d<sub>ee</sub> × H-bond bins) | ~850 ns | bin weights |
| Steered MD + Jarzynski | 32 pulls at 10 Å/ns + 32 at 100 Å/ns | Jarzynski |
| Umbrella sampling (reference) | 42 windows × 3 ns | MBAR |

## Caveats

- **Vacuum exaggerates the chemistry.** ff14SB was parameterized for solution. In vacuum,
  intramolecular H-bonds and electrostatics are exaggerated, so the compact basin is a
  prediction of this model, not an established structure.
- **The extended state is not a basin.** It is reached only under a restraint.
- **Weighted ensemble stalls.** Both WE runs stall near 21–22 Å. Extending further requires
  breaking helical H-bonds, a barrier orthogonal to d<sub>ee</sub>.
- **T-REMD and SAMS-temperature are limited on the extended side.** 700 K is not hot enough
  to unfold the vacuum helix, so their profiles are reliable only in the basins.

## Repository layout

```
sims/          simulation scripts (OpenMM + openmmtools), one run_<method>.py per method
  common.py    system, CVs, recorders, lineage log
  build.py     tleap build of the helix, minimization, equilibration, CPU benchmark
  run_all.sh   launch any subset of runs in the background
analysis/
  build_data.py   results/ -> dashboard/data/ (all plots are prebuilt Plotly specs)
  estimators.py   MBAR / WHAM / umbrella integration / stitching comparison
  shoot.py        headless-Chromium screenshots of every page (needs playwright)
dashboard/     static site: index.html, app.js, style.css, vendor/ (Plotly, 3Dmol.js), data/
gcp/           create / boot / sync scripts for a spot CPU VM on Google Cloud
results/build/ the built system (prmtop, inpcrd, equilibrated state); other results are not tracked
```

## Reproduce

```bash
micromamba create -f environment.yml -y     # openmm, openmmtools, pymbar, ambertools, ...
cd sims && source env.sh
python build.py --bench                      # build + equilibrate (~1 min), print CPU ns/day
bash run_all.sh md metad gamd smd reference reflow remd sams samsT we we2d
# ... when runs finish (about 1 h on 56 cores; the WE runs go until stopped):
cd .. && python analysis/build_data.py       # ~10 min on 2 cores
```

The system is tiny (112 atoms), so the CPU platform with one thread per process beats a GPU.
Expect about 830 ns/day per core. `gcp/create.sh` starts a spot CPU VM, and
`gcp/sync.sh push|pull` moves code and data to and from it.

## License

MIT. See [LICENSE](LICENSE). Plotly.js and 3Dmol.js in `dashboard/vendor/` are distributed under their own licenses (MIT and BSD-3-Clause).
