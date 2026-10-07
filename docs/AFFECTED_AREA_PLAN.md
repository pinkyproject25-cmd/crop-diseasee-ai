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
## Initial source audit (2026-10-07)

- PlantSeg v7 (https://zenodo.org/records/17719108) provides diseased-region
  annotations. Its record does not establish a usable whole-leaf mask or a
  clearly stated license. Do not approve it for training yet.
- CropAndWeedAndLeaf (https://zenodo.org/records/20116408) provides leaf-instance
  masks, including maize, squash, potato, and soybean. The authors state
  CC BY-NC-SA 4.0. It does not provide paired lesion masks.
- PhenoBench (https://www.phenobench.org/dataset.html) provides leaf-instance
  masks under CC BY-SA 4.0, but its sugar-beet UAV setting is a poor direct
  match for uploaded photographs of this project's target crops.

Decision: investigate CropAndWeedAndLeaf for leaf-mask training. Obtain
properly licensed target-crop photographs with leaf AND lesion masks on the
same images for affected-area evaluation. Separate datasets cannot be combined
as if their masks describe the same photograph. Keep affected area, severity,
and health score unavailable until measured validation passes.