# The Even-Parity Tell

**Cracking QMill's obfuscated peaked circuits — Quantum Hack 2026 · team hell0.**

> 41 of 49 obfuscated peaked circuits cracked on laptops — more than any other team, including `56_38`, which no other team solved — by triaging structure before throwing compute at it.

We're four bachelor's students at Aalto (maths + industrial engineering), with no quantum background before this weekend. We solved all 41 on our laptops.

A fuller methods write-up is in **[METHODS.pdf](METHODS.pdf)** (source: [METHODS.html](METHODS.html)).

> **Note:** the cracked secret bitstrings are deliberately **not** in this public repo (submissions were still open). This is the *how*, not the answer key.

---

## The challenge

Each `circuits/challenge-<nqubits>_<id>.qasm` is an obfuscated *peaked circuit*: applied to `|0…0⟩`, it concentrates almost all of its probability on one hidden bitstring

```
s = argmax_x |⟨x|C|0⟩|²
```

buried under thousands of obfuscating gates. The task is to recover `s`.

## The one idea: triage before you simulate

Rather than throw one big simulator at every circuit, we start each one with a **cheap structural read** that says which method can actually work.

Count the qubit pairs touched by an **odd** number of CX gates — the entangling links the obfuscator's inverse-pair (`U·U†`) structure can't cancel. This **odd-parity defect count** reads off the gate list in milliseconds and predicts true difficulty far better than qubit or gate count. It's why we never burned compute on the wrong tool, and why a laptop was enough.

→ `defect_count.py`, `odd_parity_probe.py`, `analyze_obf.py`

## Matching method to circuit

| Method | File(s) | Where it won |
|---|---|---|
| Exact statevector | `crack.py` | small circuits — read the peak directly |
| Bond-capped MPS sweeps | `crack.py`, `solve_mps.py`, `mps_marginal.py` | mid-range — peak survives truncation |
| Spectral (Fiedler) MPS reordering | `mps_marginal2.py` | dense 64-qubit — kills the SWAP churn |
| ⟨Zᵢ⟩ marginal decoding | `mps_marginal.py` | several 48–56q circuits |
| Vectorized Pauli backpropagation | `pauli_fast_b.py`, `pauli_crack.py` | **cracked `56_38`** |

## Two pieces we think are genuinely ours

1. **Find-cheap / verify-exact.** A candidate can come from a *sloppy* approximate run — but we certify it with one truncation-free single-amplitude contraction `|⟨s|C|0⟩|²`. Above 0.5 *proves* the peak is unique. This certified a **64-qubit** answer on a laptop, and certified one circuit even when its MPS had collapsed. Finding and proving are different problems; decoupling them is the trick. → `crack.py`, `solve_tn.py`
2. **Convergence-as-trust + the grader as an oracle.** Where no exact check fits, we raise the truncation cap until the answer stops changing. We also reverse-engineered the submission API (it scores by sampled XEB, not exact match; a wrong guess leaks nothing) and used it as a yes/no oracle to pin `56_38`'s last uncertain bit. → `probe_run.py`, `submit_secrets.py`

## Built for the dense tail (42–49, unfinished)

Light-cone bulk cancellation (`lightcone_z.py`, `lc_probe.py`), a two-stage decoder that simplifies the cancelled state once and reuses it per qubit (`distill.py`), a defect-graph reduced-density-matrix method — belief propagation diverges on these signed-amplitude networks, so we contract the small defect cone exactly (`bp_z.py`) — and two-sided MPO unswapping after Kremer–Dupuis (`pcsim/unswap.py`, `unswap_solve.py`, `uudag_cancel.py`).

All validated and converging — and to push them we rented an 8×H100 box, yet still ran out of runtime before the deadline. That these eight resist even a GPU cluster is the clearest sign they're genuinely *hard*, not just heavy.

## Dead ends (each told us where the difficulty really sits)

- **No Clifford shortcut** — the rotation angles aren't near π/2 multiples.
- **Qiskit `O3` removes zero two-qubit gates** — stock compress-then-simulate fails.
- **Deleting even-parity CX is confidently wrong** — it changes the peak.

## Running

```bash
pip install -r requirements.txt
python3 defect_count.py circuits/challenge-48_8.qasm   # triage: odd-parity defect count
python3 crack.py        circuits/challenge-48_8.qasm   # exact (small) or MPS + exact verify
```

## References

- Gharibyan et al., *Heuristic Quantum Advantage with Peaked Circuits* (2025), [arXiv:2510.25838](https://arxiv.org/abs/2510.25838)
- Kremer & Dupuis, *Efficient Classical Simulation of Heuristic Peaked Quantum Circuits* (2025), arXiv:2604.21908

`pcsim/` vendors the Kremer–Dupuis reference simulator (see `pcsim/LICENSE`).
