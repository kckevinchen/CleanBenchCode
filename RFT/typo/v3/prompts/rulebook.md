# Typo rulebook (v3, per-column audit)

You are auditing ONE column of a CSV file for typos. The column header and EVERY distinct value in the column (up to 500) are shown with their row counts. Your job is to identify every value that is a TYPO of one of the OTHER values in the same column, and to propose the intended correction.

A value is a TYPO only when, judging from that value and the column's other values, BOTH of the following hold:
  (a) you can name the specific defect, AND
  (b) the intended correction is unambiguous and requires a real       character-level edit (not merely a layout fix).
The two flavours that qualify are SPELLING and PUNCTUATION:
  - spelling: a one- or two-character distortion of a word that     plausibly belongs in this column (e.g., 'Govenor' (1 row)     next to 'Governor' (47 rows) -> spelling typo). Adjacent     DUPLICATED WORDS also qualify as spelling typos when the     de-duplicated form is the plausible intended value (e.g.,     'BUCKEYE LOCAL LOCAL SD' next to 'BUCKEYE LOCAL SD' ->     spelling typo whose correction is 'BUCKEYE LOCAL SD';     'Aircraft Aircraft' -> 'Aircraft').
  - punctuation: a wrong or missing punctuation token that     changes how the value is read (e.g., 'Address,State' ->     'Address, State'; '(575-) 894-8350' -> '(575) 894-8350').
Do NOT flag values whose only difference from the intended form is whitespace, casing, or trivial punctuation -- those are normalisation issues caught upstream, not typos. Examples that are CLEAN under this rulebook even if they look 'off':
  - 'New  York' (double space)  -- whitespace normalisation.
  - 'STRATFORD-UPON-AVON' vs 'Stratford-upon-Avon' -- casing.
  - 'Du Page' vs 'DuPage' (pure spacing) -- not a typo.
Treat everything else as CLEAN; in particular do NOT flag:
  - Recognised proper nouns, brand names, person names.
  - Fixed-length codes (zip, SKU, FIPS, ISBN, version strings).
  - Pure numeric values -- never typos by our definition.
  - Date-shaped strings -- never typos by our definition.
  - Standard idioms with elided punctuation ('New Years Day').
  - Truncated / contracted values where you cannot recover the     full form ('CLARION SAFETY SYSTE', 'Twp'). Omit these.
PROPER NOUNS, NAMES, AND ENTITIES -- EXTRA CAUTION.
Most false positives come from flagging legitimate names (people, places, organisations, products, teams, song titles, movie titles, etc.) as typos. The default for any value that looks like a named entity is CLEAN. Only flag a name as a typo when ALL of the following are true:
  (n1) the column header itself signals that values are NOT        free-form names -- i.e. the column is a controlled        vocabulary such as US state, country, month, day-of-       week, sport, department, status, category, etc., OR
       the column has a clearly canonical majority spelling        and the suspect value is the lone deviant (e.g. one        row 'Govenor' against 47 rows 'Governor');
  (n2) there is a SPECIFIC anchor value already present in        this column that differs by exactly 1-2 character edits        AND is more frequent than the suspect value; and
  (n3) the defect is a clear character-level slip (transposed        letters, dropped/duplicated letter, obviously wrong        letter) -- not a legitimate spelling variant, regional        form, foreign-language form, historical name,        transliteration, abbreviation, nickname, alias, or        stylised brand spelling.
If you cannot satisfy ALL THREE, the name is CLEAN. In particular, do NOT flag a name as a typo merely because:
  - it looks 'unusual' or you do not personally recognise it;
  - it could plausibly be a misspelling of a famous name not     present in this column;
  - its spelling differs from the form you would expect (people     and places legitimately have non-standard spellings);
  - it contains non-English letters, accents, or diacritics;
  - it appears only once in the column (rarity alone is not     evidence -- many real names appear once).
ALPHANUMERIC CODES AND IDS -- EXTRA CAUTION.
Values that mix letters and digits within a single token (e.g., 'ZUR3260', 'D5876', '2016StrawPoll', 'B12X4', 'ICC-ES-ESR3375') are codes or IDs and the default for them is CLEAN. There is no canonical spacing or punctuation schema to assume for such codes -- they legitimately appear in many internal layouts. Only flag a code as a typo when ALL of the following are true:
  (c1) there is a SPECIFIC anchor value already present in        this column that differs by exactly 1-2 character        edits AND is more frequent than the suspect value;        and
  (c2) the defect is a clear character-level slip in the        alphabetic portion (e.g., dropped or duplicated        letter) -- not a difference in spacing, casing,        hyphenation, or other punctuation between        alphanumeric segments.
In particular, do NOT flag a code merely because:
  - it concatenates letters and digits without separators     ('2016StrawPoll', 'MASSAIR');
  - it contains or omits internal whitespace between     segments ('ZUR3260 R2R M', 'PRESIDIO H S');
  - its hyphenation or punctuation differs from what you     would expect ('ESR 3375' vs 'ESR-3375').
Bias toward HIGH PRECISION: false positives cost as much as false negatives. When in doubt about a value, omit it from the typos list. The intended correction MUST be one of the OTHER distinct values shown in this column (or a near-variant of one with case/whitespace fixed); never invent a correction that has no anchor in the column.
