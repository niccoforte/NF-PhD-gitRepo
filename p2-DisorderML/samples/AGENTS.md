# Sample review pack

These are deliberately small, human-readable calculation examples requested by the user. Markdown and labelled scientific plots are the primary reading surface; a compact NPZ supports notebook replay.

- `build_examples.py` constructs labelled illustrations from the existing FCC producer rules and the actual dual feature functions. It performs no training, remote access or Abaqus export.
- Keep illustrative inputs separate from real run evidence. Never fabricate response curves, field accuracy or physical validation for the examples.
- Use coordinate units explicitly, distinguish reference geometry, initial disorder and response displacement, and identify canonical indices separately from Abaqus labels.
- Preserve representative inside/on/outside pin calculations and four/eight-strut examples. Pairwise attention remains explanatory until the user chooses that ablation.
- Real runs generate the same readable audit under their own `results/input_audit/`. Do not copy entire datasets into this folder.
- Regenerate and visually inspect plots after changing feature formulas or report rendering. Keep the two dual notebooks and README aligned.
- `field-loss-review/` contains the bounded original-INP/recorded-frame/curve-peak audit and independent HPO field-error examples. Its generators use saved validation arrays and mark rows as row indices when older artifacts lack simulation IDs. Never pair those rows across different datasets without explicit ID alignment. Numerical activity/timing proxies are not damage labels. Keep scientific caveats beside the plots.
