# Task handoffs — choose one brief

These are tracked task briefs, not executable scripts or automatic messages.
Do not ask every chat to read every file. Send the chosen chat the absolute path
of its brief, followed by: "Read this brief and applicable repository instructions.
Continue only its scope; reconcile any existing work first. Ask before new HPC
submissions, bulk processing, optimisation searches or material cleanup."

| Chat | Read this file | Owns |
| --- | --- | --- |
| Existing Improve ML Accuracy | [HANDOFF-ML-ACCURACY.md](HANDOFF-ML-ACCURACY.md) | Models/losses, incomplete experiment recovery, results, notebook diagnostics and per-test visual examples |
| New Damage variable processing | [HANDOFF-DAMAGE.md](HANDOFF-DAMAGE.md) | Element-to-strut damage export, labels and validation samples; not model integration |
| New Repo context optimisation | [HANDOFF-CONTEXT.md](HANDOFF-CONTEXT.md) | Lossless instruction/reference/status consolidation plan; no ML behaviour/storage changes |
| New Surrogate optimisation | [HANDOFF-SURROGATE-OPTIMISATION.md](HANDOFF-SURROGATE-OPTIMISATION.md) | Curve AND field-only design objectives, constraints and validated search plan; not HPO |
| Proposed Physics/strain feasibility | [HANDOFF-PHYSICS-STRAIN.md](HANDOFF-PHYSICS-STRAIN.md) | True-field strain/coupling audits before any new physics loss; no bulk exports |

The existing Tokenisation chat uses `../../code/TOKENIZATION_NEXT_STEPS.md`.
It owns motif discovery, not the predictive local-graph ablation or field-loss
implementation. Physics/strain work warrants a separate iterative task; no chat
has been created or messaged by writing its brief.

All files are beside this index in:
/Users/niccoloforte/Desktop/Code/NF-PhD-gitRepo/p2-DisorderML/HPC/DualOutputs/

For Improve ML Accuracy, append its brief path after your remaining annotations;
its own unfinished work still applies. For the other tasks, start separate chats
in this repository and provide just the corresponding path/prompt. No need to
copy the full text or to read this index first. File contents still consume
context when read; the saving comes from excluding unrelated briefs/history.

Keep model/loss ownership in Improve ML Accuracy; damage supplies a label contract,
surrogate optimisation consumes a frozen verified model, and context work changes
only approved documentation. Do not edit the same files concurrently without
coordination. Further supporting files should be read only as the task requires.

Do not delete briefs merely because they were sent. Migrate completed decisions
and outstanding actions to their proper docs, update links, then retire through
Git. The prior consolidated document is recoverable in Git history; this split
replaces it rather than keeping a duplicate.
