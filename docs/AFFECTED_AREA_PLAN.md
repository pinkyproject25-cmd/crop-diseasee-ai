# Stage 2: affected-area measurement plan

Status: design checkpoint; dataset and model are not yet approved.

## Output

For a supported, sufficiently clear photograph of one visible leaf:

affected_area_percent =
    100 × pixels in (lesion mask ∩ visible leaf mask)
        ÷ pixels in visible leaf mask

This estimates the affected portion of the photographed leaf only. It does not
measure disease prevalence in the plant or field. Classifier confidence is
never used as affected area.

If the leaf cannot be isolated, lesions cannot be measured reliably, the image
is unsupported, or validation has not covered its crop/condition, return null
and show "Unable to estimate."

## Required annotations

Training and evaluation require image-level crop/condition information plus
pixel-level masks for the visible leaf and affected regions. Healthy examples
need reviewed leaf masks and confirmation that no visible lesion is annotated.
Record how multiple leaves, occlusion, lighting, and non-disease damage are
handled. Keep related images of the same leaf in one data partition.

## Dataset investigation

PlantSeg (authors' repository: https://github.com/tqwei05/PlantSeg;
latest dataset record: https://zenodo.org/records/17719108) is a candidate for
lesion masks. Before using it, verify its exact version, license and permitted
uses, image provenance, annotation format, crop overlap, and whether it contains
whole-leaf masks. Its lesion annotations alone cannot establish the
leaf-area denominator.

Find or create a separately licensed, reviewed whole-leaf-mask dataset if
PlantSeg lacks those masks. Do not download a large dataset or begin training
until this audit is recorded.

## Evaluation before API integration

Freeze train/validation/test partitions before model selection. Use untouched
field photos with manually reviewed leaf and lesion masks for final evaluation.
Record leaf and lesion Dice/IoU, absolute error in affected-area percentage
points, per-crop performance, and failure/abstention coverage. Set numerical
acceptance thresholds before inspecting the final test results.

Severity (Low/Medium/High) requires separately justified thresholds and
validation. A health score requires its own documented meaning. Neither is
derived from classification confidence or automatically filled in.

## Current decision

No affected-area model, severity rule, or health-score rule is approved yet.
The Stage 1 API and UI correctly leave these fields unavailable.