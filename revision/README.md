# GAHIB Revision Workspace

Created: 2026-04-29

This folder is a safe revision/resubmission workspace. It preserves baseline submission material, seeds editable revision working files, and provides clearly labeled slots/placeholders for final submission-ready revision files.

## Important constraints for this pass

- The original folders (`GAHIB_submission/`, `GAHIB_submission_authored/`, `paper/`, `paper_blind/`, `cover_letter/`, and `highlights/`) were not edited.
- No LaTeX compilation was run.
- No fake final PDFs were created.
- Missing final PDFs are represented by explicit Markdown placeholder notes.
- Reviewer notes in `99_notes/potential_reviewer_comments.md` are anticipated concerns only, not actual received reviews and not point-by-point rebuttal answers.

## Layout

```text
revision/
  00_baseline_submission/
    authored_submission/   # Immutable copy of GAHIB_submission_authored/
    blind_submission/      # Immutable copy of GAHIB_submission/
  01_revision_working/
    manuscript/            # Editable manuscript seed and local figure copies
    cover_letter/          # Editable cover-letter placeholder/source
    rebuttal/              # PP rebuttal / response-to-reviewers placeholders
  02_submission_ready/
    manuscript/            # Final PDF/TEX slots and explicit missing-PDF notes
    cover_letter/          # Final cover-letter slots/source placeholder
    rebuttal/              # Final rebuttal slots and explicit missing-PDF notes
    highlights/            # Current highlights copied from authored submission package
    figures_jpg/           # Current journal-upload JPG figures copied from authored package
  99_notes/
    revision_checklist.md
    potential_reviewer_comments.md
```

## Suggested workflow

1. Treat `00_baseline_submission/` as immutable provenance.
2. Edit files only under `01_revision_working/` while preparing the actual revision.
3. Generate real final PDFs later from the revised sources.
4. Replace the Markdown placeholder notes under `02_submission_ready/` only when the real final PDFs exist.
5. Use `99_notes/potential_reviewer_comments.md` to prioritize likely revision/rebuttal work.
