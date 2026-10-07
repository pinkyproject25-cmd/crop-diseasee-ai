# User-assisted visible-area report (prototype)

The ONNX model classifies crop and disease. It does not predict lesion or leaf masks.

When the experimental automatic color estimate abstains, the Result page offers an optional marking tool on the actual uploaded image. The user outlines the visible leaf and brushes the visually affected regions. The tool counts marked pixels inside the outlined leaf; marks outside the leaf do not count. It then calculates:

- Visible affected area (%) = affected marked pixels inside leaf / outlined visible-leaf pixels × 100.
- Visual severity: Low below 10%; Medium from 10% to below 30%; High from 30% upward.
- Visibly unaffected score (%) = 100 − visible affected area (%).

These bands are demonstration rules, not agronomic disease-specific severity cutoffs. The area is based on user markings; it is neither model-segmented nor a statement about the whole plant or field. The application labels the method as user-assisted. Unknown and healthy classifications are not given disease area or severity. The user can revise markings, and the updated report is saved in browser-local history. No marked masks are uploaded to the backend.

For the review demonstration, upload an accepted diseased leaf photograph, click around its visible leaf boundary, finish the outline, brush affected patches, and calculate. Verify that the number changes with the markings; show an Unknown image remaining without disease measurements.
