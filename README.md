# Dismissed-alert audit experiments

Reproducible experiments for **Auditing Dismissed Alerts**, by Cemil Emre Yavas, Nancy R. Mead, and Lei Chen.

The implementation evaluates exact finite-cohort bounds when selected security-alert investigations complete out of order. It includes fixed-sample hypergeometric inversion, pending-verdict envelopes, prior/posterior-ratio confidence sequences, public alert-cohort replay, investigation-service simulations, and exact power calculations.

## Reproduce the results

Use Python 3.12 and the pinned numerical libraries:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --verify
```

The first run downloads the event-labeled AIT-ADS CSV archive from its pinned public source and verifies its SHA-256 hash before extraction. Internet access is needed for this download and for installing dependencies. The archive is approximately 3.4 MB; extracted CSV files occupy approximately 263 MB. Subsequent runs reuse the verified download.

The default configuration uses 12 worker processes. Adjust `workers` in `config.json` for the available CPU capacity. Numerical-library threads are limited to one per process to avoid oversubscription. An M3 Pro with 18 GB memory is sufficient. The simulation stage typically takes minutes or less after setup; exact power calculation, data preparation, dependency installation, and download add overhead. Runtime depends on processor, cache state, and network access.

`--verify` compares every reproduced CSV against `reference_results/` with a relative and absolute floating-point tolerance of 1e-12, and checks the exact-validation summary. Random streams use a fixed master seed and deterministic task-specific seeds. Worker count does not change the task seeds. Runtime and operating-system metadata in `run_metadata.json` describe the current execution and are recorded separately from numerical verification.

## Contents

- `config.json`: experimental parameters, audit budgets, time grids, sample counts, and master seed.
- `src/audit.py`: scalar exact inversion, confidence-sequence endpoints, and uniform binary sample generation.
- `src/prepare_data.py`: pinned data retrieval, checksum verification, held-out signature scoring, and chronological cohort construction.
- `src/run_experiments.py`: vectorized lookup tables, parallel audit trials, cohort replay, and service simulations.
- `src/power.py`: exact budget, nonresponse, and benign-dilution operating characteristics.
- `tests/test_exact.py`: exhaustive finite-population and completion-envelope checks, ordered-path confidence-sequence checks, and independent lookup comparisons.
- `data/manifest.json`: source commit, download address, file size, and cryptographic checksum.
- `reference_results/`: reference numerical tables and validation summaries.
- `run.py`: complete reproduction and verification entry point.

## Experiment definitions

### Controlled audits

Sixteen conditions cross four malicious counts with four malicious-to-benign delay ratios. Each condition uses 50,000 uniform audits from a population of 4096 alerts, selecting 256. The `failure` column in `synthetic.csv` counts trials with an upper bound below the true malicious count at any inspected time up to the specified time. `mean_upper` is the mean prevalence upper bound. The confidence level is 95% per cohort.

Five methods are compared on matched samples and delays. `Completed fixed` applies a fixed-sample bound to completed cases. `Completion-order CS` consumes verdicts in completion order using a Beta(0.5,24.5) working prior. `Random-prefix CS` consumes the longest resolved prefix of the original random sample using Beta(0.5,0.5). `Risk-informed prefix CS` uses Beta(0.5,24.5) with that valid original prefix. `Pending envelope` counts unresolved selected items as potentially malicious. The first two methods are diagnostic ablations; completion selection can invalidate them. The prefix and envelope methods satisfy the stated design assumptions.

### Public cohort replay

An event label other than `-` marks an attack-related record. A leave-one-scenario-out smoothed signature score creates the dismissal sets. Only the other seven scenarios contribute to each score. Dismissed records are sorted by timestamp and split into cohorts of at most 4096, keeping the final partial cohort. Repeated audits condition on these fixed cohort labels and simulate verdict delays. They measure audit operating characteristics on a generated testbed, not independent enterprise deployments. `dataset_summary.csv` and `cohorts.csv` contain the source counts; `replay.csv` contains time-specific conditional audit outcomes.

### Service simulation

`queue.csv` contains per-trace, per-batch outcomes for shared priority service and work-conserving reserved audit service. Both policies receive matched arrivals and external delays. `certificate_latency` is measured in simulated slots; an empty value denotes right censoring at the replay horizon. Batches in one arrival trace share a queue. `backlog_at_end` is the final priority-work backlog of that trace, repeated on its batch rows. The modeled audit cohorts are benign so this experiment isolates certificate availability.

### Exact sensitivity

`frontier.csv` gives minimum all-benign sample sizes for count and rate requirements. `budget_power.csv` averages exact certification probabilities over the 202 fixed public cohorts, using either per-cohort or simultaneous error allocation. `nonresponse.csv` varies permanent benign nonresponse. `dilution.csv` keeps 20 malicious alerts fixed while adding benign records. These tables are exact distribution calculations rather than Monte Carlo estimates.

## Data provenance and licensing

The public AIT Alert Data Set is maintained by the Austrian Institute of Technology:

- Dataset: https://zenodo.org/records/8263181
- Event-labeled CSV release: https://github.com/ait-aecid/alert-data-set
- Source revision and file checksum: `data/manifest.json`
- Data description: M. Landauer et al., “Introducing a New Alert Data Set for Multi-Step Attack Analysis,” CSET 2024, https://doi.org/10.1145/3675741.3675748.

The upstream repository carries the GNU General Public License version 3. Downloaded upstream materials retain their upstream licensing terms. The original experiment implementation in this repository is distributed under the MIT License in `LICENSE`.

## Contact

Cemil Emre Yavas: cemil@fairesresearch.org
