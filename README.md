# AI Annotation & Served Inference

**Pay humans to label support tickets, or automate it?** A retrieval-plus-LLM annotation
pipeline over **13,083 real customer-support messages across 77 intents**, with a four-arm
ablation, calibration, a derived abstention gate, distillation, a cost frontier, and a
deployed API.

**~2,900 lines · 45 tests passing · verified on CPython 3.13.9, MiniLM, flan-t5-base**

Every number was measured on a run and written to `outputs/results.json`. Nothing is
estimated.

---

## The finding: the language model does not earn its place

The pipeline was built the way these are usually built — a dense retriever shortlists
candidate intents, then a language model re-ranks them. The four-arm ablation asks whether
that second stage is worth anything.

| arm | what it isolates | accuracy | LM forward passes |
|---|---|---:|---:|
| **A** | retriever only, no LLM at all | **0.8920** | **0** |
| B | LLM only, retriever ranking discarded | 0.6040 | 4,036 |
| C | fusion of both | 0.8920 | 4,036 |
| D | LLM with 3 in-context examples *(300-row subset)* | 0.8967 | 703 |

**The retriever alone beats the language model by 28.8 points.** And the fusion sweep is
the cleanest statement of it — accuracy rises *monotonically* as weight shifts away from
the LM:

| α (weight on retriever) | 0.0 | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 | 0.9 | **1.0** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| accuracy | 0.6412 | 0.6887 | 0.7188 | 0.8450 | 0.8962 | 0.8988 | 0.9000 | **0.9025** |

**The optimum is α = 1.0 — zero weight on the language model.** Every increment of LM
influence makes it worse. Arm D confirms it from another direction: few-shot prompting
scored 0.8967 against the retriever's 0.9300 *on the same rows*.

So the LLM is **not on the serving path**, and it is not in the Docker image either.
Shipping a 250M-parameter model the service never calls would be carrying the opposite of
the conclusion.

**Why the retriever is so strong here:** BANKING77 has ~130 labelled examples per intent,
and the intents are phrasing-driven. A kNN vote over sentence embeddings is close to the
ideal method for that — and flan-t5-base has no idea what
`card_payment_wrong_exchange_rate` means as a category, only as words. This is a result
about *this problem shape*, and the honest scope is stated in the limits.

---

## Calibration: the model was badly *under*-confident

The abstention gate routes on confidence, so that confidence has to mean something.

| | before | after |
|---|---:|---:|
| ECE | 0.2459 | **0.0233** |
| mean confidence | — | — |
| fitted temperature | — | **0.1730** |

**A 90.5% reduction in calibration error from one scalar.** And T = 0.173 is well *below*
1, meaning the model was systematically under-confident — the opposite of the usual
cross-entropy failure mode. That is a consequence of the scoring method: similarity-weighted
kNN votes are normalised into a near-flat distribution, so the raw confidence is
compressed toward uniform and needs sharpening rather than softening.

Temperature scaling cannot change a single prediction — dividing every score by the same
positive number preserves the argmax exactly — so accuracy is untouched and only the
confidences move.

## The abstention gate was derived, not chosen

Picking 0.8 because it looks sensible is how these systems end up either flooding the
review queue or shipping errors. Instead: state the precision the business requires, then
find the **lowest** threshold achieving it, which maximises coverage subject to that
guarantee.

At **95% precision**: threshold **0.7093**, coverage **86.9%**.
**196 of 1,500 messages go to a human. The other 1,304 are answered automatically.**

The trade is reported as a table, not a single number, because raising the requirement
costs coverage and that cost is what the conversation should be about.

## Distillation: a 29,645-parameter linear model retains 98.9%

The student is trained on the **teacher's** labels, not the gold labels — that is what
distillation means, and it is what you would actually have in production, where gold labels
for new data do not exist.

| | parameters | accuracy | vs teacher |
|---|---:|---:|---:|
| teacher (retriever) | 22.7M + a 10,003-vector index | 0.8920 | — |
| neural student (MLP) | 118,349 | 0.8787 | −0.0133 |
| **linear student** | **29,645** | **0.8820** | **−0.0100** |

**The linear model beats the MLP** despite having 4× fewer parameters — the extra capacity
buys nothing, which is exactly why the linear control was included. And the learning curve
is still climbing at 100% of the training data, so the ceiling here is *labels*, not
architecture. Those two diagnoses have opposite actions, and only measuring tells you which
one you have.

## The cost frontier: three live options, three dominated

Priced per 1,000 items on the same basis, including the human review each option's gate
sends back:

| option | $/1k | accuracy | coverage | effective | |
|---|---:|---:|---:|---:|---|
| **retriever only** | **$0.0002** | 0.8920 | 100% | 0.8920 | on the frontier |
| distilled student | $0.0003 | 0.8820 | 100% | 0.8820 | dominated |
| LM only | $0.0084 | 0.6040 | 100% | 0.6040 | dominated |
| retriever + LM re-rank | $0.0086 | 0.8920 | 100% | 0.8920 | dominated |
| **retriever + abstention gate** | **$10.89** | 0.9502 | 87% | **0.9567** | on the frontier |
| **human annotation** | **$83.33** | 1.0000 | 0% | 1.0000 | on the frontier |

Only three options are live. **The LM re-rank costs 40× the retriever and adds nothing** —
dominated on both axes, and no rational buyer picks it.

The interesting part is what the frontier says about the *shape* of the decision: automated
inference is essentially free (a fraction of a cent per thousand), so **cost is not the
real trade — accuracy is.** That is precisely why the gated option exists. It is 54,000×
more expensive than the raw retriever and still 7.6× cheaper than pure human annotation,
and it buys the precision back with review.

---

## The pipeline

```
message
   │
   ├─ 1. RETRIEVE   MiniLM embeddings, kNN vote over 10,003 labelled messages
   │                recall@8 = 0.990  ← the ceiling on everything downstream
   │
   ├─ 2. SCORE      flan-t5-base scores the 8 candidates      [MEASURED, THEN REJECTED]
   │
   ├─ 3. FUSE       α sweep → α = 1.0, i.e. discard stage 2
   │
   ├─ 4. CALIBRATE  temperature scaling, ECE 0.2459 → 0.0233
   │
   ├─ 5. ABSTAIN    threshold derived at 95% precision → 86.9% coverage
   │
   └─ 6. SERVE      FastAPI, ~5ms p50, no language model in the image
```

**recall@8 = 0.990 is reported first and deliberately.** It bounds every downstream stage:
if the right label is not in the shortlist, nothing later can recover it. A two-stage system
whose first stage quietly loses 12% of the answers looks like a second-stage problem
forever.

## Layout

| path | lines | what |
|---|---:|---|
| `src/` | ~1,300 | retriever, LM scorer, fusion, ablation, calibration, gate, student, distillation, cost |
| `serve/` | ~380 | FastAPI, three health probes, schemas |
| `dashboard/` | 271 | the live classifier and threshold UI |
| `tests/` | ~470 | 45 tests |
| `data/` + `infra/` + CI | ~430 | fetcher, Dockerfile, compose, Fly, GitHub Actions |

## Run it

```bash
make setup
make all          # data → pipeline → tests   (~90s)
make serve        # http://localhost:8600
```

**Type a banking question and watch it classify or abstain.** The UI shows the nearest
training messages that produced the answer — since the system *is* a kNN vote, those
neighbours are the actual explanation, not a post-hoc rationalisation. Then move the
threshold slider and watch coverage and precision trade against each other.

```bash
make quick        # a 400-row pipeline run
make docker
```

No API key is needed for anything: MiniLM and flan-t5-base both run locally on CPU.

## Known limits

- **The LM result is about this problem shape, not about language models.** BANKING77 has
  ~130 examples per intent and is phrasing-driven, which is close to ideal for kNN. With
  10 examples per class, or unseen intents, or a task needing reasoning rather than
  matching, the ranking would very likely reverse.
- **One language model.** flan-t5-base is 250M parameters. A frontier model would score far
  better — and would also cost far more than $0.0084/1k, which is the comparison that
  actually matters and is not made here.
- **The cost assumptions are assumptions.** $10/hour and 30 seconds per item are stated at
  the top of `src/cost.py`; change them and every figure moves. They are labelled, not
  hidden.
- **Human accuracy is defined as 1.0 by construction**, because the gold labels *are* the
  human labels. Real annotator agreement on 77 fine-grained intents is meaningfully below
  100%, which flatters the human baseline on quality and understates how competitive
  automation is.
- **The abstention gate is measured on the same test set the threshold was derived from.**
  A held-out threshold-fitting split would be more rigorous; with 3,080 test rows the
  effect is small but it is real.
- **The learning curve is still rising**, so the student is label-limited. More teacher
  labels are cheap to generate and were not.

See `DECISIONS.md` for why each choice was made and what was rejected.
