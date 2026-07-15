# Typo rulebook (v1)

A cell is a TYPO only when, judging from the cell value and column header alone, you can name the specific defect AND the intended correction is unambiguous. Three flavours qualify:
  - spelling: a one- or two-character distortion of a word that     plausibly belongs in this column (e.g., 'Acomodate' ->     'Accommodate'; 'Pilot Mountatin' -> 'Pilot Mountain').
  - whitespace: internal repeated spaces, embedded tab or newline     (e.g., 'New  York'). Leading/trailing whitespace is stripped     in preprocessing and must NOT be flagged.
  - punctuation: missing comma+space between tokens, doubled     commas, stray hyphens (e.g., 'Address,State' ->     'Address, State'; '(575-) 894-8350' -> '(575) 894-8350').
Use AMBIGUOUS when the cell looks suspicious but the rulebook cannot uniquely resolve it from cell + header alone:
  - Truncated values whose full form you cannot recover     ('CLARION SAFETY SYSTE', 'Education in Unorganized Terr').
  - Contraction-style abbreviations ('Twp', 'Gl Hearn Limited').
  - Multiple equally plausible corrections.
Use CLEAN for everything else, especially:
  - Recognised proper nouns, brand names, person names.
  - Fixed-length codes (zip, SKU, FIPS, ISBN, version strings).
  - Pure numeric values -- never typos by our definition.
  - Date-shaped strings -- never typos by our definition.
  - Standard idioms with elided punctuation ('New Years Day').
When in doubt, prefer AMBIGUOUS over TYPO.
