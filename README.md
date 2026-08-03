# Supplementary materials

This archive accompanies the experience-report paper "Long-Horizon Autonomous
Architecture Research with a Language-Model Agent: A Behavioural Case Study".

It contains everything needed to audit the per-hypothesis trace described in
the paper and to regenerate every figure and table, together with a compiled
`appendix.pdf` that records the full implementation stack (Appendix A) and the
complete per-hypothesis record (Appendix B). The full training code (model
implementation, training driver, and per-run experiment configurations) is not
included in this bundle and will be released under a permanent identifier upon
publication.

## Contents

```
.
|-- README.md                                  (this file)
|-- appendix.pdf                               compiled Appendix A (implementation
|                                              details, instruction documents,
|                                              autonomy failure modes, costs) and
|                                              Appendix B (full per-hypothesis
|                                              record as a single table)
|-- instructions/
|   |-- RESEARCH_PROGRAM.md                    full research-programme document
|   |-- RESEARCH_BRIEF.md                      shorter agent-facing brief that
|   |                                          points to RESEARCH_PROGRAM.md
|   |                                          and adds per-phase constraints
|   |-- AGENT_INSTRUCTIONS.md                  one-page workflow companion
|   |-- TOOLING.md                             tooling-configuration document
|   |                                          (MCP servers, Git CLI, etc.)
|   `-- empirical_laws_synthesis.md            agent-authored synthesis read
|                                              alongside the log in long sessions
|-- log/
|   |-- research_log.md                        full per-hypothesis log
|   |                                          (one entry per hypothesis in the
|   |                                          seven-field template of Appendix A)
|   `-- hypotheses.csv                         parsed projection of the log
|                                              used to generate every figure
|-- outputs/
|   `-- architecture_evolution.md              agent-authored architectural
|                                              specification, one of the
|                                              by-products of the loop
|-- figures/
|   |-- parse_notes.py                         log -> CSV
|   `-- build_figures.py                       CSV -> figure PDFs
|-- configs/
|   |-- phase1_champion.yaml                   architecture spec, end of Phase 1
|   |-- phase2_champion.yaml                   architecture spec, end of Phase 2
|   `-- phase3_baseline.yaml                   architecture spec, Phase 3 baseline
`-- slurm/
    |-- train_1gpu.slurm.template              single-GPU submission template
    |-- train_2gpu.slurm.template              two-GPU submission template
    |-- job_watcher.sh.template                control-hand-off watcher
    `-- submit_and_wait.sh.template            alternative submission utility
```

## Reproducing the analysis

To regenerate `log/hypotheses.csv` from the research log:

```
cd figures
python parse_notes.py
```

To regenerate the analysis figures:

```
cd figures
python build_figures.py
```

This writes `fig_trajectory.pdf`, `fig_success_rate.pdf`, `fig_attribution.pdf`,
`fig_hypothesis_mix.pdf`, `fig_cross_scale.pdf`, and `fig_bias_decomp.pdf` to the
same directory (the paper uses a subset of these). The full per-hypothesis table
in Appendix B is similarly regenerable from the CSV.

## Reproducing a training run

The training code is not included in this bundle. The configurations in
`configs/` and the Slurm templates in `slurm/` document the exact
hyperparameters and submission protocol used for each phase. With the training
code (released upon publication), the workflow to reproduce a hypothesis is:

```
sbatch --export=ALL,CFG=configs/phaseX_champion.yaml,RUN_TAG=repro \
       slurm/train_2gpu.slurm.template
```

The Slurm template expects to be customised at the top for the target site
(partition name, accelerator constraint, conda environment path, data path).

## Map from supplementary files to the paper

| File | Referenced in |
| --- | --- |
| `appendix.pdf`                                             | full implementation + per-hypothesis record |
| `instructions/RESEARCH_PROGRAM.md`, `RESEARCH_BRIEF.md`, `AGENT_INSTRUCTIONS.md` | System design; Appendix A |
| `instructions/TOOLING.md`                                  | Appendix A (agent and tooling) |
| `instructions/empirical_laws_synthesis.md`                 | System design; case study |
| `log/research_log.md`                                      | Quantitative analysis; Appendix A, B |
| `log/hypotheses.csv`                                       | All figures; Appendix B |
| `outputs/architecture_evolution.md`                        | Case study (outputs of the loop) |
| `figures/parse_notes.py`, `build_figures.py`               | All figures |
| `configs/phase*_*.yaml`                                    | Case study; Appendix A |
| `slurm/*.template`                                         | System design; Appendix A |

## Notes on portability

Site-specific identifiers (user IDs, cluster paths, partition names, and
accelerator generations) have been replaced with neutral placeholders so that
the templates are portable. The agent, command-line harness, tool servers, and
experiment-tracking service are named in the paper and in `appendix.pdf`. The
original artefacts (with identifiers restored and the full training code) will
be released under a permanent identifier upon publication.
