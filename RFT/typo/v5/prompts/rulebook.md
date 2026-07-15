# Typo rulebook

You audit ONE CSV column. Input: the column header and every distinct value
(up to 500) with row counts. Output: every value that, judged ON ITS OWN as
a string, is a typo, paired with its intended correction.

**Per-cell judgement.** Do not require an in-column anchor. Each value
is evaluated on its own merits as a string. The column listing is context
only (truncation patterns, controlled vocabularies, proper-noun caution).

---

## 1. What counts as a TYPO

A value is a typo when **all three** hold:
- **(a) specific defect** -- you can name it in one short sentence;
- **(b) unambiguous fix** -- a competent reader, looking only at this
  value, agrees on the same correction;
- **(c) real edit** -- a character-level change OR a non-trivial
  word-boundary spacing change. NOT just padding whitespace, casing, or
  trivial punctuation (sec.2).

Three flavours qualify.

### `spelling`
One- or two-character distortion of a real word -- transposed adjacent
letters, dropped letter, duplicated letter, or obviously wrong letter --
where the correction is a frequent English word or domain term.

| Value          | Correction     | Defect          |
|----------------|----------------|-----------------|
| `Recieved`     | `Received`     | i-before-e      |
| `Adress`       | `Address`      | dropped letter  |
| `Manger`       | `Manager`      | dropped letter  |
| `Seperate`     | `Separate`     | wrong letter    |
| `Mozarella`    | `Mozzarella`   | dropped letter  |
| `Independance` | `Independence` | wrong letter    |

**Adjacent duplicated words** also qualify when the de-duped form is
plausible: `Aircraft Aircraft` -> `Aircraft`,
`Sales Sales Manager` -> `Sales Manager`,
`Northern Northern District` -> `Northern District`.

### `punctuation`
Wrong or missing punctuation that changes how the value reads AND has a
single obvious fix.

| Value               | Correction        | Defect                       |
|---------------------|-------------------|------------------------------|
| `(555-) 123-4567`   | `(555) 123-4567`  | stray hyphen in area code    |
| `St.. Louis`        | `St. Louis`       | doubled period               |

### `spacing`
Non-trivial whitespace defect that changes how the value tokenises into
words -- missing or extra word-boundary space inside what should be one
orthographic unit, where the canonical form is unambiguous. **Not in
scope here**: padding whitespace (double/triple spaces, leading/trailing
whitespace) -- those stay CLEAN under sec.2.

| Value             | Correction         | Defect                          |
|-------------------|--------------------|---------------------------------|
| `web site`        | `website`          | canonical form is one word      |
| `e mail`          | `email`            | canonical form is one word      |
| `data base`       | `database`         | canonical form is one word      |
| `peanutbutter`    | `peanut butter`    | canonical phrase has a space    |

Apply only when canonical spacing is unambiguous from the value itself.
When the column shows both spaced and fused forms as legitimate stylistic
variants, treat as CLEAN.

---

## 2. Always CLEAN

These are never typos, however off they look. Each item names the
deterministic upstream rule that handles it -- the LLM must not waste
precision budget on cases a regex already fixes.

| CLEAN case                                     | Handled by                                          |
|------------------------------------------------|-----------------------------------------------------|
| `New  Haven`, `New   York`                     | collapse internal whitespace runs                   |
| ` Apple`, `Apple `                             | strip leading/trailing whitespace                   |
| `NEWCASTLE-UPON-TYNE` vs `Newcastle-upon-Tyne` | case canonicalisation policy                        |
| `Apples,Oranges & Pears`, `Sales,Marketing`    | insert space after comma before letter/digit        |
| `Sales-Marketing`                              | hyphen-spacing policy (deterministic, not LLM)      |
| `Acme Holdings Inc`                            | append `.` to closing `Inc`/`Co`/`Ltd` if required  |
| `555 123-4567` vs `555-123-4567`               | phone-number format canonicalisation                |
| `2024-25***`                                   | strip trailing footnote markers (`*`, `dagger`, etc.) |

Any difference that is **only** about padding whitespace, case, or a
single comma / period / asterisk / dash is CLEAN. Word-boundary spacing
defects inside a word that change tokenisation are NOT covered here --
those are `spacing` typos per sec.1.

**Fixed-width truncations / contracted values.** If many distinct values
cluster at the same maximum length AND many end mid-word, the column was
exported through a fixed-width field. All such mid-word-ending values
are CLEAN. Rule of thumb: **>=25% of distinct values within 1 char of
max length -> treat the whole column as truncated; omit all truncation-
shaped values**. Examples: `WESTERN INDUSTRIAL SUPPLI`,
`XX-DEPTNAME-OPERATIONS-GENERA`, `XX-PROGNAME22-OPERATING EX`,
`PURCHASING & SUPPLI`, `Regional Community School Of Newpor`,
`TRANSPORT OF EQUIPMENT PARTS AND TOOLI`. Standard contractions
(`Twp`, `St.`, `Dept`) are also CLEAN.

**Standard idioms with elided punctuation.** Holiday/observance names,
possessive titles, and idiomatic place names routinely written without
their dictionary apostrophe stay CLEAN: `Workers Day`, `Founders Day`,
`Settlers Day`, `Farmers Market`, `Mens Room`, `Womens Room`.

**Other CLEAN categories.**
- **Pure numeric values** (`30.01` vs `30`, `1.50` vs `1.5`).
- **Date-shaped strings** -- handled by outlier detection, not typo
  detection (`1919` vs `2019` is CLEAN here).
- **Fixed-length identifier codes** -- ZIP, SKU, FIPS, ISBN, versions,
  account codes.
- **Recognised proper nouns** -- people, places, orgs, products, teams,
  titles. See sec.3.
- **Alphanumeric codes/IDs** -- mixed letters+digits in one token. See sec.4.

---

## 3. Proper nouns and entities -- EXTRA CAUTION

Default for any value that looks like a named entity is **CLEAN**. Flag a
proper-noun value only when **all** hold:
- **(n1)** clear character-level slip (transposed/dropped/duplicated/
  wrong letter) in a token that is a real English word or widely
  recognised place / organisation noun (e.g. `Country` used for
  `County`);
- **(n2)** correction is a frequent dictionary form or widely recognised
  entity -- NOT a rare name you happen to guess;
- **(n3)** the defect is NOT a legitimate variant, regional / foreign /
  historical form, transliteration, abbreviation, nickname, alias, or
  stylised brand spelling.

Do NOT flag a name merely because it looks unusual, could plausibly be a
misspelling of a famous name, differs from your expected spelling, uses
non-English letters/diacritics, or appears only once in the column.

---

## 4. Alphanumeric codes/IDs -- EXTRA CAUTION

Values that mix letters and digits within one token are codes; default
is **CLEAN**. Example shapes: `XYZ1234`, `A567B`, `C99D2`,
`PROD-A-12345`, `QQQ7890`.

Flag only when the defect is a character-level slip in the alphabetic
portion that yields an obviously intended English word
(e.g. `PROFESIONAL` -> `PROFESSIONAL`). NEVER flag differences in
spacing, casing, hyphenation, or punctuation between segments
(`XYZ1234WIDGET`, `XYZ1234 R2 M` vs `XYZ1234R2M`,
`PROD A 12345` vs `PROD-A-12345`).

---

## 5. Output

Return a single JSON object -- no Markdown fences, no surrounding text.

```
{
  "typos": [
    {
      "value":      "<typo-bearing value, verbatim>",
      "correction": "<intended canonical form>",
      "flavour":    "spelling | punctuation | spacing",
      "rationale":  "<one short sentence naming the defect>"
    },
    ...
  ]
}
```

Empty list when there are no typos. Every value NOT in the list is
treated as CLEAN.

**Bias toward HIGH PRECISION**: false positives cost as much as false
negatives. When in doubt, omit.

---

## 6. Self-check before emitting each entry

Drop the entry if any answer is "no":
1. Did I name a specific defect in one short sentence?
2. Would a competent reader, looking only at this value, agree on the
   same correction?
3. Is the edit a real character-level change OR a non-trivial
   word-boundary spacing change (not padding/casing/trivial
   punctuation)?
4. Is the value safe from sec.2 always-clean categories (truncation,
   idiom, numeric, date, fixed-length code)?
5. If it looks like a proper noun or a code, do **all** sec.3 / sec.4
   conditions hold?

