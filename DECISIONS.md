# Decisions — AI Annotation & Served Inference

Every non-obvious choice, the alternatives, and why they lost.

---

## D-01 · BANKING77, and why fine-grained intents matter

**Chose:** 13,083 real support messages across 77 near-synonymous intents.

**Why not a 5-class dataset:** every method looks good on an easy problem, and the result
would say nothing about which to use. Here `card_arrival` vs `card_delivery_estimate` and
`top_up_failed` vs `top_up_reverted` are genuinely hard — a human annotator has to think,
and a keyword matcher is hopeless.

**Why the size matters:** ~130 examples per intent is exactly the regime where "pay humans
or automate?" is a live question rather than an obvious one.

**`csv.reader`, not a manual split.** Several messages contain commas inside quoted fields
("I still have not received my new card, I ordered over a week ago."). A naive split
truncates them and shifts the label column — silently, and in a way that looks like model
error later.

## D-02 · A retrieval shortlist before any LLM call

**Chose:** kNN over embeddings produces 8 candidates; the LLM scores only those.

**Why:** scoring all 77 labels costs 77 forward passes per message. Shortlisting to 8 costs
8 — a 9.6× reduction.

**And the price is measured, not assumed.** recall@8 = 0.990 is reported first, because it
is the **ceiling on everything downstream**. If the right label is not in the shortlist, no
amount of downstream cleverness recovers it. A two-stage system whose first stage quietly
loses 12% of answers will look like a second-stage problem forever.

## D-03 · kNN voting, not a class centroid

**Chose:** each of the 24 nearest neighbours votes for its own label, weighted by
similarity.

**Why not a centroid per class:** intents like `card_arrival` have genuinely multi-modal
phrasings — "where is my card", "still waiting", "ordered two weeks ago". A single centroid
averages them into something that matches none of them well.

**Why similarity-weighted rather than counted:** a neighbour at cosine 0.91 is much stronger
evidence than one at 0.42. Counting them equally lets a cluster of weak matches outvote one
strong one.

## D-04 · Mean pooling, not [CLS]

MiniLM's sentence-level behaviour comes from a **mean-pooling training objective**. [CLS] is
not trained to carry it. Using [CLS] with this model family is a common mistake that
silently degrades every number downstream and produces no error.

**Masked** mean pooling specifically: padding tokens must not contribute, or short texts
batched with long ones get their embeddings diluted by however much padding they received.

## D-05 · Constrained scoring, not free generation

**Chose:** ask the LM for `log P("yes" | message, candidate)` per candidate.

**Why not generate the label:** it means parsing free text, handling near-misses
("card arrival" vs `card_arrival`), and having **no confidence number at the end**.
Constrained scoring yields a real distribution over exactly the candidates, which
calibration and abstention both require.

**Length normalisation is not optional.** Log-likelihood is a sum over tokens, so a longer
verbalisation is mechanically less likely. Without dividing by token count the model
systematically prefers `atm_support` over `card_payment_wrong_exchange_rate` for reasons
unrelated to the message. This is the single most common bug in constrained-decoding
classifiers.

**Labels are verbalised** (`card_payment_fee_charged` → "card payment fee charged"). The raw
identifier never appears in natural text, so the model has no learned representation of it.

## D-06 · Both signals z-scored before fusion

**Chose:** standardise the retriever's and the LM's scores *within each example* before
mixing.

**Why:** they live on completely different scales with completely different spreads. Adding
them raw means whichever happens to have larger variance dominates, regardless of which is
more informative — and α would then not mean what it appears to mean.

**Within-example, not global:** the comparison that matters is between candidates for the
*same* message. A global standardisation would let an easy message with high scores
everywhere shift a hard one's ranking.

**Zero variance is handled explicitly** — all candidates identical means no information, and
dividing by ~0 manufactures enormous differences out of floating-point noise.

## D-07 · Report the whole α sweep, not the argmax

**Chose:** accuracy at eleven values of α, including both endpoints.

**Why the endpoints:** they *are* the individual systems, so the sweep answers "does fusion
beat either component?" in one table rather than as a claim.

**Why the whole curve:** it shows whether the optimum is a broad plateau (robust) or a
narrow spike (probably overfitting to this test set). Here it is monotone to α = 1.0, which
is about as unambiguous as this can get: **every increment of LM weight makes it worse.**

## D-08 · Arm D on a subset, with a two-step winner rule

**Chose:** evaluate few-shot on 300 rows, and compare it against the other arms *restricted
to those same rows*.

**Why a subset:** few-shot prompts are ~3× longer and ~3× slower. Running it on the full set
would triple the ablation's runtime to answer one question.

**Why the two-step rule:** comparing a subset score against full-set scores compares
different quantities. D is eliminated on its own rows first; the winner is then chosen among
A/B/C on the full set. Skipping that step is how ablations produce winners that are
artifacts of unequal evaluation.

## D-09 · The expensive LM pass is cached to disk

The scoring pass is the costliest thing in the project. Re-running it to change a
downstream threshold would make iteration impossible, so it is cached against a signature
of (model, test size, shortlist size) and invalidated when any of those change.

## D-10 · Temperature scaling by golden-section search

**Chose:** a derivative-free line search over one scalar.

**Why not gradient descent:** there is nothing to tune — no learning rate, no stopping
criterion — and one scalar does not justify an optimiser. NLL is smooth and unimodal in T on
this interval, so the search converges in a handful of evaluations.

**Why T is fitted only on examples whose gold label is in the shortlist:** for the rest there
is no correct class to assign probability to, and including them pushes T toward flattening
every distribution to hedge against the unreachable.

**The measured result was T = 0.173 — under-confident**, which is the opposite of the usual
cross-entropy failure mode, and a direct consequence of normalising similarity-weighted
votes into a near-flat distribution.

## D-11 · The abstention threshold is derived from a precision target

**Chose:** state the required precision; find the **lowest** threshold achieving it.

**Why lowest:** any higher threshold also meets the target but auto-accepts fewer items,
costing more human review for the same guarantee. Maximum coverage subject to the constraint
is the objective.

**Why this framing at all:** the business does not have an opinion about confidence scores.
It has an opinion about how often an automatic answer may be wrong. Inverting the question
is what makes the threshold defensible instead of arbitrary.

**Unachievable targets are reported honestly** rather than returning a threshold that
silently does not do what it claims.

**The sweep across targets is reported**, because raising precision from 95% to 99% has a
coverage cost and that cost is the actual conversation.

## D-12 · The student is trained on TEACHER labels

**Chose:** distil from the pipeline's own predictions, not from gold.

**Why:** training on gold is ordinary supervised learning, not distillation — and it does
not reflect production, where gold labels for new data do not exist. Training on teacher
labels has a real ceiling (the student cannot learn what the teacher got wrong), and whether
it lands near that ceiling is the measurement.

**A linear control is included**, and it *won*: 0.8820 against the MLP's 0.8787, with 4×
fewer parameters. Without the control, the MLP would have shipped and the extra capacity
would have looked necessary.

**A learning curve is reported** because it distinguishes "the student is too small" from
"the student needs more labels" — diagnoses with opposite actions. It is still rising at
100%, so this one is label-limited, and teacher labels are cheap to generate.

## D-13 · Every cost assumption is stated at the top of one file

**Chose:** `HUMAN_SECONDS_PER_ITEM`, `HUMAN_HOURLY_USD`, `COMPUTE_HOURLY_USD` and the
measured throughputs, all declared together with their rationale.

**Why:** a cost model with magic numbers scattered through it cannot be argued with, and a
cost model nobody can argue with is a cost model nobody should believe. Anyone who disagrees
changes a constant and re-runs.

**Machine throughputs are measured on this machine**, not quoted from a spec sheet — 20
items/second for the LM, 850 for embedding, both observed during the run.

**Fixed costs amortise over volume**, so the index build does not make the first thousand
items look expensive forever.

## D-14 · Gated options are charged for the review they generate

An automated option that routes 60% of traffic to a person is **not 99% cheaper**. Every
option's price includes `(1 − coverage) × human_cost`, which is the only way the comparison
is like-for-like.

## D-15 · Six decimal places on cost

**REGRESSION.** Rounding to cents collapsed every automated option to `$0.00`, and the ratio
against humans then printed as **83,330,000,000× cheaper**. The whole point of a frontier is
comparing options that differ by orders of magnitude, so the precision has to survive that
range. A test now asserts sub-cent costs stay non-zero.

## D-16 · Dominated options are named, not just ranked

**Chose:** report the Pareto frontier and list what is dominated.

**Why:** a ranked list implies every row is a live candidate. Three of six options here are
beaten on *both* cost and quality — including the LM re-rank — and no rational buyer picks
them. Saying so is more useful than a sorted table.

## D-17 · The language model is not in the serving image

**Chose:** the Docker image installs MiniLM and not flan-t5-base.

**Why:** the ablation measured the LM at 0.6040 against the retriever's 0.8920, and the
fusion optimum put zero weight on it. Shipping a 250M-parameter model the service never
calls would be carrying the opposite of the conclusion — and it would add ~1GB and a slower
cold start for nothing.

**This is what it means to act on an ablation.** An ablation that does not change what
ships was a decoration.

## D-18 · The threshold is read from an artifact, not hard-coded

`serve/app.py` reads `abstain_threshold` from `artifacts/serving.json`, where the pipeline
wrote the value it derived. A constant in the service would drift away from the analysis
that justified it the first time either changed — and nothing would fail.

**The same applies to the temperature.** The threshold was derived on *calibrated*
confidences, so serving raw ones would silently move the operating point while every number
in the README stayed the same.

## D-19 · Three health endpoints

- **live** — a failure KILLS the container, so it must never touch the model; otherwise a
  model problem becomes a restart loop.
- **ready** — a failure removes the instance from the load balancer without killing it, and
  returns **503**, because 200 with `ready: false` keeps traffic arriving.
- **startup** — suppresses liveness during a slow boot so a 20-second index load is not
  repeatedly killed at 10.

## D-20 · The UI shows the nearest training messages

**Chose:** `explain=true` returns the neighbours that produced the vote.

**Why this is not a bolt-on:** the system *is* a kNN vote, so those neighbours are the
literal computation, not a post-hoc rationalisation of a black box. It is the rare case
where the honest explanation is also the simplest one, and it is worth exploiting rather
than replacing the model with something less interpretable.
