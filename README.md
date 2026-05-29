# branch-predictor-sim

Cycle-accurate branch predictor simulator implementing four distinct predictors — Bimodal, GShare, Tournament, and Perceptron — behind a shared vtable interface. Evaluates each against a 2-million-branch synthetic trace and reports mispredictions per kilo-instruction (MPKI).

---

## Predictor Hierarchy

```
                    BranchPredictor (vtable interface)
                    ┌──────────────────────────────────┐
                    │  predict(self, pc) → bool        │
                    │  update (self, pc, outcome)      │
                    │  destroy(self)                   │
                    │  name   → const char*            │
                    └──────────────────────────────────┘
                             ▲           ▲
               ┌─────────────┤           ├─────────────┐
               │             │           │             │
          Bimodal         GShare    Tournament     Perceptron
         (2-bit sat.)   (XOR hash)  (meta-pred.)  (weight table)
```

All predictors are allocated on the heap and accessed exclusively through the vtable pointer. `main.c` drives all four through a single `bp_evaluate()` call in a loop — adding a fifth predictor requires only implementing the vtable and zero changes to the harness.

---

## Bimodal Predictor

A table of 2-bit saturating counters indexed directly by the low bits of the PC.

```
  PC[13:0]  (14-bit index)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  table[16384]  (2-bit saturating counters)                  │
│                                                             │
│   00 ──→ strongly not-taken                                 │
│   01 ──→ weakly not-taken                                   │
│   10 ──→ weakly taken          ← threshold: predict taken   │
│   11 ──→ strongly taken                                     │
└─────────────────────────────────────────────────────────────┘
      │
predict: counter ≥ 2  →  taken
update:  taken  → sat_inc(counter, 3)
         not    → sat_dec(counter)
```

---

## GShare Predictor

Uses a Global History Register (GHR) XOR-hashed with the PC to index a counter table. Captures inter-branch correlation that Bimodal cannot.

```
  PC[13:0]                GHR[13:0]
      │                       │
      └──────── XOR ───────────┘
                    │
                    ▼  (14-bit index)
          ┌──────────────────┐
          │  table[16384]    │  (2-bit saturating counters)
          └──────────────────┘
                    │
              predict / update

After each branch:
  GHR = (GHR << 1 | outcome) & mask_14bit
```

---

## Tournament Meta-Predictor

Selects between Bimodal and GShare using a per-PC choice table. When the two predictors disagree, the meta-predictor biases toward whichever was correct.

```
  PC[13:0]
      │
      ▼
┌─────────────────────────────────────┐
│  choice[16384]  (2-bit counters)    │
│                                     │
│  ≥ 2 → prefer GShare               │
│  < 2 → prefer Bimodal              │
└─────────────────────────────────────┘
      │
      ├─ use_gshare? → gshare_predict(pc)
      └─ else        → bimodal_predict(pc)

Update rule (only when predictors disagree):
  gshare correct  → sat_inc(choice)   (lean toward gshare)
  bimodal correct → sat_dec(choice)   (lean toward bimodal)
Both sub-predictors are always updated regardless.
```

---

## Perceptron Predictor

A linear threshold machine (Jiménez & Lin, 2001). Maintains `T` hash tables of integer weights. Prediction is the sign of the dot product between weight rows and the global history vector.

```
PC  +  Global History Register (62 bits)
│
├── Hash 0: (pc >> 0)  ^ 0×2654435761  → idx_0 → weight_0 × h[0]
├── Hash 1: (pc >> 3)  ^ 1×2654435761  → idx_1 → weight_1 × h[8]
├── Hash 2: (pc >> 6)  ^ 2×2654435761  → idx_2 → weight_2 × h[16]
│   ...
└── Hash 7: (pc >> 21) ^ 7×2654435761  → idx_7 → weight_7 × h[56]
                                                       │
                                              y = Σ sign(h[i]) × w[i]
                                                       │
                                             predict:  y ≥ 0  →  taken
```

### Training Rule

```
θ = floor(1.93 × history_len + 14)   (training threshold)

if (prediction ≠ outcome) OR (|y| ≤ θ):
    for each table h:
        w[h] += (outcome ? +1 : -1) × sign(history_bit[h])
        w[h]  = clamp(w[h], −127, +127)
```

Training only fires when the prediction was wrong **or** the confidence is low (|y| ≤ θ). This avoids weight saturation on easy branches.

---

## Synthetic Trace Generator

```
64 branch sites, three classes:

  sites  0–50  (80%)   Strongly biased: 90% taken
                        → easy for all predictors

  sites 51–60  (15%)   Loop pattern: taken 8 times then not-taken
                        → periodic, GShare/Tournament advantage

  sites 61–63   (5%)   Near-random: 52% taken
                        → hard for all predictors
                        → Perceptron shows most resilience
```

XORshift-64 PRNG; seed is configurable. Branch sites are generated once and reused to simulate realistic PC clustering.

---

## Evaluation Pipeline

```
for each branch in trace (2 000 000):
  ┌────────────────────────────────────────┐
  │  predict(bp, pc)   →  guess            │
  │  if guess ≠ outcome: misses++          │
  │  update(bp, pc, outcome)               │
  └────────────────────────────────────────┘
                     │
             MPKI = misses / (trace_len / 1000)
```

Each predictor is evaluated independently on an identical trace from the same seed, so comparisons are fair.

---

## API

```c
// Factory functions
BranchPredictor *bimodal_create(int table_bits);
BranchPredictor *gshare_create(int table_bits, int history_len);
BranchPredictor *tournament_create(int meta_bits,
                                   BranchPredictor *primary,
                                   BranchPredictor *secondary);
BranchPredictor *perceptron_create(int num_tables, int history_len,
                                   int table_entries);

// Vtable dispatch (inline)
bool        bp_predict(BranchPredictor *bp, uint64_t pc);
void        bp_update (BranchPredictor *bp, uint64_t pc, bool outcome);
void        bp_destroy(BranchPredictor *bp);
const char *bp_name   (const BranchPredictor *bp);

// Trace
BranchTrace    *trace_generate(size_t length, uint64_t seed);
void            trace_destroy(BranchTrace *t);

// Evaluation
PredictorResult bp_evaluate(BranchPredictor *bp, const BranchTrace *trace);
void            bp_print_result(const PredictorResult *r);
```

---

## Build

```sh
gcc -O2 -std=c11 branch_predictor.c main.c -o branch-predictor-sim
```

---

## Sample Output

```
branch-predictor-sim: bimodal / gshare / tournament / perceptron

  trace length    : 2000000 instructions

  Bimodal              MPKI = 133.56
  GShare               MPKI = 140.39
  Tournament           MPKI = 135.30
  Perceptron           MPKI = 124.12
```

Perceptron wins because the near-random class responds better to long-history correlation than to short counters. GShare occasionally aliases unrelated branches in the XOR-hashed table, hurting it relative to Bimodal on this particular trace.

---

## File Structure

```
branch-predictor-sim/
├── branch_predictor.h    ← vtable definition, factory prototypes, result types
├── branch_predictor.c    ← all four predictor implementations, trace generator
└── main.c                ← configuration constants, evaluation loop
```
