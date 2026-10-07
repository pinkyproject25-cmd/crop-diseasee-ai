# Stage 2 knowledge prototype

Status: three source-linked English entries drafted; agricultural and
localisation review required before treating the catalogue as complete.

The backend selects an entry by the exact classifier label only after an
accepted diseased classification. Unknown, blurry, healthy, and unlisted
classes receive no disease-specific knowledge. Entries describe *typical*
symptoms and general next steps, not observations extracted from the image.
The UI links directly to the reference for each populated entry.

| Exact label | Reference | Scope |
| --- | --- | --- |
| Apple___Apple_scab | [UC IPM](https://ipm.ucanr.edu/agriculture/apple/apple-scab/) | Signs, causal agent, wet conditions, orchard sanitation |
| Grape___Black_rot | [Cornell CALS](https://cals.cornell.edu/news/2014/03/grapes-101-managing-black-rot) | Leaf lesions, berry mummies, inoculum, sanitation |
| Potato___Late_blight | [UMN Extension](https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/late-blight) | Signs, pathogen, damp conditions, scouting |

These US extension references are starting points. The project owner should
review terminology and recommendations for the deployment region with a
qualified agricultural expert. This prototype avoids pesticide product,
dosage, and timing instructions. A single leaf photo cannot verify symptoms
elsewhere on a plant. Translation and speech endpoints still require their
separate provider credentials and validation.

Next: review the three entries; extend exact-label coverage systematically;
record an owner/reviewer and review date for each entry. In parallel, follow
the paired leaf-and-lesion mask gate in AFFECTED_AREA_PLAN.md. Do not fill
affected area, severity, or health score from this catalogue.
