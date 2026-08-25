# AI Slop Detector: Fine-Tuning Task Plan

This is the working plan for Phase 4 and the later integration phases.

The repository remains the source of truth for code, configuration, experiments,
and documentation. Heavy data inspection and Transformer training are submitted
from the local repository to Kaggle through the Kaggle CLI. Kaggle is the remote
compute backend, not the home of our source code.

The goal is to build a binary text classifier that predicts whether a passage is
human-written or AI-generated, while understanding and measuring every important
part of the training process.

---

## 0. Current project state

### Completed

- Project structure and Git repository.
- Dataset downloaders and RAID parsing utilities.
- Local RAID sample downloads.
- Dynamic PMI vocabulary extractor.
- Burstiness, RTTR, and Shannon entropy features.
- Unified feature pipeline.
- XGBoost fallback classifier.
- Serialized PMI vocabulary and XGBoost model.
- Local Kaggle CLI authentication.
- Kaggle RAID audit kernel and first downloaded audit report.
- Standard-library RAID manifest builder with unit tests.

### Current artifacts

- models/ai_vocabulary_weights.json
- models/xgb_baseline.json
- data/raid_train_sample.csv — local labeled sample, ignored by Git.
- data/raid_sample.csv — local unlabeled inspection sample, ignored by Git.

### Limitations of the local samples

The local 50 MB training sample is useful for parser and feature smoke tests, but
it is not a representative final training set:

- It is strongly imbalanced toward AI records.
- It is dominated by the first domain blocks in the CSV.
- It contains limited model and attack coverage.
- A byte-range CSV slice may end in a partial final record.
- It must not be treated as the final RAID evaluation set.

### Not yet implemented

- Full RAID training data contract implementation.
- Grouped train/validation split.
- Transformer classifier.
- Fine-tuning script.
- LoRA/PEFT training.
- Transformer checkpoint retrieval.
- Local Transformer inference.
- XGBoost/Transformer cascade.
- Web UI.

---

## 1. Architecture and responsibilities

The final detector is expected to have two model paths:

~~~text
Input text
    |
    +--> Fast feature path
    |       PMI + burstiness + RTTR + entropy
    |       XGBoost fallback
    |
    +--> Deep text path
            tokenizer
            Transformer encoder
            classification head
            AI probability
~~~

The models solve different problems:

- XGBoost is fast, CPU-friendly, and explainable.
- The Transformer reads token-level context and learns patterns that four
  handcrafted features cannot represent.
- The Transformer must be measured against XGBoost on the same held-out records.
- The cascade is a later integration step; each model must first be correct and
  independently measurable.

---

## 2. Local-to-Kaggle workflow

### Principle

All source code stays local. Selected jobs are submitted to Kaggle:

~~~text
Local code
    |
    | kaggle kernels push
    v
Kaggle remote job
    |
    | checkpoints, metrics, and reports
    v
Local artifacts/
~~~

### Local responsibilities

- Write and review source code.
- Run unit tests and smoke tests.
- Use the local RAID sample for fast debugging.
- Prepare Kaggle kernel files and metadata.
- Submit jobs using the Kaggle CLI.
- Download and inspect outputs.
- Commit source code, configurations, reports, and reproducibility metadata.

### Kaggle responsibilities

- Access the full or streamed RAID data.
- Provide GPU compute.
- Run audit and training jobs.
- Save checkpoints and metric reports.
- Produce reproducible output artifacts.

### Setup checklist

- [x] Authenticate the local Kaggle CLI.
- [ ] Verify the active Kaggle account.
- [ ] Confirm the selected accelerator is available.
- [ ] Create a local Kaggle kernel directory.
- [ ] Create kernel-metadata.json.
- [ ] Submit a minimal GPU health-check job.
- [ ] Download the first output artifact locally.

Suggested verification commands:

~~~bash
.venv/bin/kaggle datasets list
.venv/bin/kaggle kernels list
.venv/bin/kaggle kernels status USERNAME/KERNEL-SLUG
.venv/bin/kaggle kernels output USERNAME/KERNEL-SLUG -p artifacts/
~~~

Credentials must never be committed. Kaggle credential files and directories must
remain ignored by Git.

---

## 3. Phase 4A: RAID data audit

The first Phase 4 job is an audit, not model training.

Proposed directory:

~~~text
kaggle/raid_audit/
    raid_audit.py
    kernel-metadata.json
~~~

### Audit objectives

- Confirm that the remote job can access the intended dataset.
- Print the exact schema.
- Verify the text column.
- Verify the label construction rule.
- Count human and AI rows.
- Count generator models.
- Count domains.
- Count decoding strategies.
- Count repetition-penalty settings.
- Count adversarial attacks.
- Measure text lengths.
- Check missing and empty generations.
- Check duplicate IDs.
- Check repeated source_id values.
- Check repeated adv_source_id values.
- Save a compact report.

### Data source decision

The Kaggle job must use one verified source:

1. A verified Kaggle Dataset handle attached through kernel metadata.
2. The official Hugging Face RAID dataset loaded from Kaggle with the datasets
   library and streaming enabled.

We must not assume that a third-party Kaggle RAID upload is identical to the
official dataset. Its files, labels, domains, and attack coverage must be checked.

### Audit success criteria

- The job runs remotely.
- GPU visibility is confirmed when GPU is enabled.
- The dataset opens without schema errors.
- At least one human and one AI record are observed.
- The columns needed for training are present.
- The report is downloaded into artifacts/.
- The report records dataset source and version.

---

## 4. Phase 4B: Training data contract

Before writing a training loop, define exactly what one training example means.

### Required fields

At minimum:

- text: the generation field.
- label: 0 for human and 1 for AI.
- source_id: original source grouping.
- adv_source_id: adversarial grouping when available.
- model: generator or human.
- domain: source genre/domain.
- attack: attack name or none.
- decoding: generation strategy.
- repetition_penalty: generation setting.

### Initial label rule

~~~text
model == human  -> label 0
otherwise       -> label 1
~~~

This rule must be verified against the loaded data rather than assumed.

### Classification decision

The first detector is a binary classifier:

~~~text
0 = human-written
1 = AI-generated
~~~

All non-human values in the `model` field belong to the AI class. Values such
as `gpt2`, `mpt`, `mistral`, and `llama-chat` are not separate target classes
for the first model. They are metadata used to construct a diverse split and
to report per-generator evaluation metrics.

A separate generator-identification experiment may be added later, but it is
not the main detector objective.

### Text cleaning

Cleaning must be conservative because unusual text can be a useful robustness
signal.

Allowed initial processing:

- Normalize obvious parser artifacts.
- Remove empty examples.
- Preserve punctuation.
- Preserve capitalization.
- Preserve whitespace information for a separate robustness experiment.
- Preserve Unicode characters for adversarial testing.

Do not aggressively lowercase, strip punctuation, or remove unusual characters
before the robustness experiments.

### Records versus chunks

A document may produce multiple token chunks. Keep:

- record_id: original RAID record.
- chunk_id: individual training window.

All chunks from one source must remain in the same split.

---

## 5. Phase 4C: Correct data splitting

Random row splitting is not sufficient because related rows can share source text
or adversarial lineage.

### Grouping rules

Random row splitting is not sufficient. We must prevent the same underlying
source, generated variant, attack variant, or text chunk from appearing in
different partitions.

The manifest builder should create a leakage group from the connected
`source_id` and `adv_source_id` relationships:

1. Records sharing a `source_id` belong to one group.
2. Records sharing an `adv_source_id` belong to one group.
3. If either identifier links two records, both records stay together.
4. A stable record identifier is used only when no source grouping exists.

This is stronger than choosing only one ID as the group key. After splitting,
we must explicitly verify that neither `source_id` nor `adv_source_id` overlaps
between partitions.

All token chunks and attack variants from one group must stay together.

### Initial split

~~~text
Training:   80%
Validation: 10%
Held-out test: 10%
~~~

The exact ratios may change after the audit, but the grouping rule must remain.

The first pilot should contain approximately 10,000 examples with roughly equal
class counts, for example 5,000 human and 5,000 AI. The audit sample was
approximately 8% human and 92% AI, so it should not be used as the only
training distribution for the first experiment.

Balancing is performed by selecting groups while targeting class counts; rows
must not be sampled independently if that would split a group. We will later
evaluate both on this balanced pilot and on a naturally distributed test set.

### Evaluation partitions

Report results separately for:

- Clean text, where attack is none.
- Each adversarial attack.
- Each domain.
- Each generator model.
- Each decoding strategy.
- Human samples.
- Text-length buckets.

For the AI class, report generator-specific metrics without changing the binary
target. This tells us whether the detector generalizes across generator
families instead of merely recognizing one model's writing style.

### Leakage checks

- No source_id overlap between splits.
- No adv_source_id overlap between splits.
- No exact-text duplicates across splits.
- PMI vocabulary is fitted on training data only.
- Any normalization statistics are fitted on training data only.

---

## 6. Phase 4D: Tokenization and sequence length

### Concepts to learn

- A tokenizer maps text to token IDs.
- A token is not always one word.
- input_ids identify tokens.
- attention_mask identifies real tokens versus padding.
- Special tokens mark sequence boundaries.
- max_length limits context.
- Truncation discards text beyond the limit.
- Padding makes a batch rectangular.

### Initial strategy

Start with a fixed maximum length such as 512 tokens. It is easier to debug and
gives a controlled baseline.

For each example, record:

- Original character length.
- Original token length.
- Whether truncation occurred.
- Number of resulting chunks.

### Long-text experiments

After the first classifier works, compare:

1. First-window truncation.
2. Fixed-size non-overlapping chunks.
3. Overlapping chunks.
4. Mean probability aggregation.
5. Maximum probability aggregation.
6. A long-context model.

Long context must be measured rather than assumed to be better.

---

## 7. Phase 4E: Transformer model selection

The first model should be an encoder used for classification, not a generative
language model.

Candidate sequence:

~~~text
First baseline: DeBERTa-v3-small or DeBERTa-v3-base
Second experiment: ModernBERT-base
~~~

### Selection criteria

- Fits available Kaggle GPU memory.
- Has a compatible tokenizer.
- Supports sequence classification.
- Has a clear model license.
- Can be saved and loaded locally.
- Has acceptable inference latency.
- Provides meaningful improvement over XGBoost.

### Model components

The classifier consists of:

1. Pretrained Transformer backbone.
2. Pooling or classification representation.
3. Optional dropout.
4. Linear classification head.
5. Two output logits: human and AI.

The logits are converted to probabilities with softmax.

---

## 8. Phase 4F: Understand and test the training loop

Before relying on a high-level Trainer abstraction, understand its underlying loop.

### Forward pass

~~~text
input_ids + attention_mask
    -> Transformer
    -> hidden representations
    -> classification head
    -> logits
~~~

### Loss

For two-class classification:

~~~text
loss = CrossEntropyLoss(logits, labels)
~~~

The loss is low when the correct class receives high probability and high when the
model is confidently wrong.

### Backpropagation

Each training step:

1. Clear old gradients.
2. Run the forward pass.
3. Calculate loss.
4. Compute gradients with backpropagation.
5. Optionally clip gradients.
6. Update trainable weights with the optimizer.
7. Update the learning-rate scheduler.

### Hyperparameters to record

- Learning rate.
- Batch size.
- Gradient accumulation steps.
- Number of epochs.
- Warmup ratio or steps.
- Weight decay.
- Dropout.
- Maximum sequence length.
- Evaluation frequency.
- Checkpoint frequency.
- Random seed.
- Precision mode.

---

## 9. Phase 4G: Tiny overfit test

Before training thousands of samples, train on a tiny fixed subset.

### Procedure

- Select 32–100 examples.
- Use a fixed random seed.
- Train long enough to overfit.
- Observe training loss and training accuracy.

### Expected result

The model should nearly memorize the tiny training set.

If it cannot, investigate:

- Incorrect labels.
- Tokenization errors.
- Wrong label column.
- Frozen parameters.
- Incorrect loss inputs.
- Broken optimizer.
- Incorrect attention masks.
- Data-collator problems.
- Learning rate that is too small.

This test verifies the pipeline; it does not measure generalization.

---

## 10. Phase 4H: First real fine-tuning experiment

Use a balanced and grouped subset after the audit.

### Initial experiment

~~~text
Data:              approximately 10,000 examples
Classes:           approximately balanced
Sequence length:  512 tokens
Model:             selected encoder baseline
Epochs:            1–2
Evaluation:        grouped validation split
Checkpointing:     save best validation checkpoint
Precision:         use mixed precision after correctness is verified
~~~

### Required outputs

Each run must save:

- Model checkpoint.
- Tokenizer.
- Training configuration.
- Dataset split metadata.
- Random seed.
- Training loss history.
- Validation metrics.
- Confusion matrix.
- Evaluation summary.
- Git commit identifier when available.

---

## 11. Phase 4I: LoRA and parameter-efficient fine-tuning

Introduce LoRA after the normal classification path is understood.

### Concept

Full fine-tuning updates all model weights. LoRA freezes the pretrained backbone
and injects small trainable low-rank matrices into selected layers.

~~~text
W' = W + delta_W
delta_W = B A
~~~

The matrices A and B are much smaller than the original weight matrix W.

### Why use LoRA

- Lower GPU memory use.
- Fewer trainable parameters.
- Faster experiments.
- Smaller adapter artifacts.
- Easier preservation of the original base model.

### LoRA decisions to record

- Target modules.
- Rank r.
- LoRA alpha.
- LoRA dropout.
- Bias handling.
- Whether the classification head remains fully trainable.
- Adapter save path.

### Required comparison

Compare full fine-tuning and LoRA on the same split:

- Accuracy.
- F1.
- Recall.
- False-positive rate.
- Training time.
- GPU memory.
- Checkpoint size.
- Inference behavior.

LoRA is an engineering choice, not an automatic accuracy improvement.

---

## 12. Phase 4J: Evaluation protocol

Accuracy alone is insufficient for this detector.

### Required metrics

- Accuracy.
- Precision.
- Recall.
- F1 score.
- ROC-AUC.
- PR-AUC.
- Confusion matrix.
- False-positive rate.
- False-negative rate.

### Threshold selection

The model outputs a probability, but the final label depends on a threshold.

The default threshold of 0.5 is only a starting point. Select the threshold on
validation data, for example to satisfy a chosen false-positive constraint, then
freeze it before final testing.

### Required reports

Produce overall reports and breakdowns by:

- Domain.
- Generator.
- Attack.
- Decoding.
- Text length.
- Human versus AI class.

### Baseline comparison

Evaluate Transformer and XGBoost on exactly the same held-out records.

The comparison must answer:

- Does the Transformer improve overall performance?
- Does it improve attack robustness?
- Does it reduce false positives?
- Is the improvement worth the inference cost?
- Does XGBoost remain useful as a fallback?

---

## 13. Phase 4K: Scaling experiments

Scale only after the small pilot is correct.

Suggested progression:

~~~text
32–100 examples    pipeline overfit test
1,000 examples     training smoke test
10,000 examples    first real baseline
50,000 examples    stronger pilot
100,000+ examples  scaling experiment
Full streamed data production run
~~~

Each scale-up must preserve:

- The same split methodology.
- The same evaluation protocol.
- Versioned configuration.
- Reproducible seeds.
- Saved metrics.

Avoid changing model, data size, sequence length, and evaluation method all at
once. Otherwise we cannot explain why a result changed.

---

## 14. Phase 4L: Checkpoint export and local inference

After a successful Kaggle run:

1. Download the model or adapter.
2. Download the tokenizer.
3. Download the configuration.
4. Download the threshold and metrics.
5. Store them under a model artifact directory.
6. Load them through a local inference class.
7. Run local smoke tests.

Proposed structure:

~~~text
models/
    transformer_detector/
        config.json
        tokenizer/
        adapter/
        threshold.json
        metrics.json
~~~

The local inference API should eventually expose:

~~~python
result = detector.predict(text)

{
    "ai_probability": 0.87,
    "label": 1,
    "model": "transformer",
    "threshold": 0.50
}
~~~

---

## 15. Phase 5: Cascade integration

The cascade is designed only after independent model evaluation.

### Candidate cascade

~~~text
Input text
    |
    v
XGBoost feature path
    |
    +-- high-confidence human -> return fast result
    |
    +-- high-confidence AI -> return fast result
    |
    +-- uncertain -> Transformer
~~~

Confidence thresholds must be learned from validation data. XGBoost confidence
must not be assumed to be calibrated.

### Cascade metrics

Report:

- Overall accuracy.
- Overall F1.
- False-positive rate.
- Percentage routed to Transformer.
- Average latency.
- CPU-only latency.
- Transformer invocation rate.
- Performance on difficult and adversarial examples.

### Hybrid feature experiment

Only after the pure Transformer baseline works, test whether existing features add
value:

~~~text
Transformer representation
        +
normalized PMI/burstiness/RTTR/entropy
        ->
combined classification head
~~~

This is a separate experiment from the normal Transformer classifier.

---

## 16. Learning checklist

Before implementing each component, be able to explain:

- Why the data split is valid.
- What a label represents.
- Why grouped splitting prevents leakage.
- What a tokenizer produces.
- Why padding and attention masks are needed.
- What logits represent.
- How cross-entropy loss is calculated.
- What a gradient means.
- What the optimizer changes.
- Why learning rate matters.
- What an epoch and batch are.
- Why validation data is separate from training data.
- Why a tiny overfit test is useful.
- What full fine-tuning changes.
- What LoRA freezes and trains.
- Why probabilities are not automatically calibrated.
- Why accuracy is insufficient.
- Why false positives matter for AI-text detection.
- Why attack/domain breakdowns matter.
- Why XGBoost remains useful after adding a Transformer.

---

## 17. Definition of done for Phase 4

Phase 4 is complete only when:

- [ ] Local Kaggle submission workflow works.
- [ ] RAID source and version are recorded.
- [ ] Dataset schema is audited.
- [ ] Labels are verified.
- [ ] Grouped splits are implemented.
- [ ] Leakage checks pass.
- [ ] Tokenization is tested.
- [ ] Tiny overfit test passes.
- [ ] First Transformer baseline trains remotely.
- [ ] Validation metrics are saved.
- [ ] Per-domain/model/attack metrics are saved.
- [ ] Threshold selection is documented.
- [ ] Checkpoint and tokenizer can be downloaded.
- [ ] Local inference reproduces Kaggle predictions.
- [ ] Transformer is compared with XGBoost.
- [ ] LoRA is evaluated rather than assumed.

---

## 18. Immediate next tasks

The audit and classification decision are now complete. The next
implementation sequence is:

1. Implement a local manifest builder for the verified RAID schema.
2. Convert `model` into the binary human/AI label and validate the rule.
3. Remove empty records and exact duplicate texts deterministically.
4. Build leakage groups using connected `source_id`/`adv_source_id` relationships.
5. Create grouped 80/10/10 train, validation, and held-out test manifests.
6. Select an approximately balanced 10,000-example pilot without breaking groups.
7. Verify split invariants and save class/model/attack/domain distributions.
8. Submit the manifest audit to Kaggle and inspect the downloaded report.
9. Implement tokenizer loading and the 32–100-example tiny overfit test.
10. Only after the tiny test succeeds, create the first Transformer fine-tuning job.

### Current concrete task

The first local implementation of step 1 is now in
`src/data/raid_manifest.py`, with tests in `tests/test_raid_manifest.py`. It
reads labeled RAID records and writes JSONL manifests containing the original
fields plus:

- `label`
- `split`
- `group_id`
- `record_id`
- `text_length`

The builder is deterministic given a random seed and fails loudly when split
leakage checks fail. It does not tokenize or train a model.

The next step is to run this logic against a real RAID CSV slice, then add a
Kaggle streaming adapter so the same manifest logic can operate on the full
labeled RAID training split without requiring a local 16 GB download.
