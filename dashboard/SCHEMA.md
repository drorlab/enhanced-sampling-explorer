# Dashboard data schema (produced by analysis/build_data.py)

All paths relative to `dashboard/data/`. Units: d_ee in Angstrom, F in kcal/mol, time in ns unless named `_ps`.

## index.json
```js
{
  generated: "ISO time",
  system: { name, n_atoms, blurb },
  methods: [                       // sidebar order
    { key: "md"|"metad"|"smd"|"we"|"remd"|"sams"|"gamd"|"reference",
      name, color, blurb,          // blurb: 1-2 sentences, what the method does
      fe_how,                      // 1 sentence, how FE is estimated
      total_ns, dF, dF_err,        // dF = F(30 A) - F(15 A) or null
      needs_cv: true|false }
  ],
  lineage: {                       // per method key
    <key>: {
      nodes: [ { id, label, kind, t_ps, traj: "<traj key>"|null,
                 strip: null | { n: [...], dmax: [...] } } ],   // strip: WE per-iteration walkers / max d_ee
      edges: [ [fromId, toId], ... ]
    }
  },
  comparison: {                    // the summary view
    panels: [ Panel, ... ]         // same Panel shape as below
  }
}
```
Node kinds: build, minimize, equil, prod, cmd, pull_slow, pull_fast, sampler, window, pull,
replica, stage1, stage2, iterations (WE, has strip).

## <key>.json  (one per method, and "comparison" lives in index)
```js
{
  key, name, color,
  stats: [ { value: "20 ns", label: "simulated" }, ... ],
  trajectories: [ { key, label, group } ],     // viewer picker; group = chip row label
  default_traj: "<traj key>",
  panels: [ Panel, ... ]                       // rendered in order in a 2-column grid
}
```
`Panel = { id, title, hint, wide: bool, data: [plotly traces], layout: {plotly layout overrides} }`
The frontend merges `layout` over its base layout and calls `Plotly.react`.

## traj/<traj key>.json + traj/<traj key>.bin
```js
{ n_frames, n_atoms, t_ns: [...], d_ee: [...], n_hb: [...], frame_label: [...] | null,
  bin: "<traj key>.bin", title }
```
`.bin` = little-endian float32, shape [n_frames, n_atoms, 3], Angstrom, aligned to the helix
backbone. Atom order = `topology.pdb`.

## topology.pdb
Single model, all atoms (incl. H), residues ACE ALA×10 NME.
