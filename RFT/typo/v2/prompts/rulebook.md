# Typo rulebook (v2, binary)

A cell is a TYPO only when, judging from the cell value and column header alone, BOTH of the following hold:
  (a) you can name the specific defect, AND
  (b) the intended correction is unambiguous and requires a real       character-level edit (not merely a layout fix).
The two flavours that qualify are SPELLING and PUNCTUATION:
  - spelling: a one- or two-character distortion of a word that     plausibly belongs in this column (e.g., 'Acomodate' ->     'Accommodate'; 'Pilot Mountatin' -> 'Pilot Mountain').
  - punctuation: a wrong or missing punctuation token that     changes how the value is read (e.g., 'Address,State' ->     'Address, State'; '(575-) 894-8350' -> '(575) 894-8350').
Do NOT flag values whose only difference from the intended form is whitespace, casing, or trivial punctuation -- those are normalisation issues caught upstream, not typos. Examples that are CLEAN under this rulebook even if they look 'off':
  - 'New  York' (double space)  -- whitespace normalisation.
  - 'STRATFORD-UPON-AVON' vs 'Stratford-upon-Avon' -- casing.
  - 'Du Page' vs 'DuPage' (pure spacing) -- not a typo.
Use CLEAN for everything else, especially:
  - Recognised proper nouns, brand names, person names.
  - Fixed-length codes (zip, SKU, FIPS, ISBN, version strings).
  - Pure numeric values -- never typos by our definition.
  - Date-shaped strings -- never typos by our definition.
  - Standard idioms with elided punctuation ('New Years Day').
  - Truncated / contracted values where you cannot recover the     full form ('CLARION SAFETY SYSTE', 'Twp'). The benchmark may     score these as 'ambiguous' but you must commit to CLEAN here.
When in doubt, prefer CLEAN over TYPO.
