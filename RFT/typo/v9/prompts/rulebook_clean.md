# Typo rulebook (clean / spec-derived)

**Provenance.** Every rule and example below is derived ONLY from the
abstract `typo` definition in `prompts/error_types.yaml` and from
universally-known, dictionary/style-guide misspellings. No string in
this file was taken from any evaluation set, gold benchmark, or labeled
CSV. This rulebook is meant to be FROZEN before it is scored: it must
not be edited in response to benchmark results (see RFT.md D10/D14/F5).
Its purpose is an out-of-distribution-honest GPT-5.5 detector prompt.

You audit ONE CSV column. Input: the column header and every distinct
value (up to 500) with row counts. Output: every value that, judged ON
ITS OWN as a string, is a typo, paired with its intended correction.

**Per-cell judgement.** Each value is evaluated on its own merits as a
string. The column listing is context only — it can reveal a controlled
vocabulary, a house abbreviation style, or a truncation pattern, but it
is never required to justify a flag.

---

## 1. What counts as a TYPO

A value is a typo when **all three** hold:

- **(a) specific defect** — you can name it in one short sentence;
- **(b) unambiguous fix** — a competent reader, looking only at this
  value, agrees on the same correction;
- **(c) real edit** — a character-level change, OR a word-boundary
  spacing change that alters how the value tokenises into words. NOT
  merely padding whitespace, casing, or trivial punctuation (sec. 2).

Three flavours qualify.

### `spelling`
A one- or two-character distortion of a real word — transposed adjacent
letters, a dropped letter, a duplicated letter, or an obviously wrong
letter — where the correction is a common English word or a generic
domain term.

| Value          | Correction     | Defect          |
|----------------|----------------|-----------------|
| `Recieved`     | `Received`     | i-before-e      |
| `Seperate`     | `Separate`     | wrong letter    |
| `Adress`       | `Address`      | dropped letter  |
| `Occured`      | `Occurred`     | dropped letter  |
| `Definately`   | `Definitely`   | wrong letter    |
| `Maintainance` | `Maintenance`  | extra letter    |

**Adjacent duplicated words** also qualify when the de-duped form is
plainly intended: `Total Total Amount` → `Total Amount`.

### `punctuation`
Wrong or missing punctuation that changes how the value reads AND has a
single obvious fix (e.g. a doubled period `word..` → `word.`). Apply
only when the corrected punctuation is unambiguous from the value alone.

### `spacing`
A word-boundary whitespace defect that changes how the value tokenises
into words — a missing or extra space inside what should be one
orthographic unit — where the canonical form is unambiguous.

| Value          | Correction      | Defect                       |
|----------------|-----------------|------------------------------|
| `web site`     | `website`       | canonical form is one word   |
| `e mail`       | `email`         | canonical form is one word   |

Apply only when canonical spacing is unambiguous from the value itself.
When a column shows both spaced and fused forms as legitimate stylistic
variants, treat as CLEAN.

---

## 2. Always CLEAN

These are never typos, however off they look. Each is a difference a
deterministic upstream step already normalises, so do not spend
precision budget on them.

- **Padding whitespace** — internal double/triple spaces, leading or
  trailing spaces.
- **Casing only** — upper/lower/title-case differences.
- **A single trivial-punctuation difference** — one comma, period,
  asterisk, or dash that does not change a word's letters.
- **Pure numeric values** — `12.50` vs `12`, `1.50` vs `1.5`.
- **Date-shaped strings** — handled by outlier detection, not here.
- **Fixed-length identifier codes** — ZIP, SKU, FIPS, ISBN, version
  strings, account codes.
- **Standard abbreviations / acronyms / initialisms** — short legitimate
  forms such as country/state codes, `Inc`, `Ltd`, `Ave`, `St`, `Dr`,
  `Mt`. A short value is CLEAN when it reads as a clean abbreviation,
  initialism, all-caps acronym, or consonant/first-syllable contraction.
  Flag a short value ONLY when it is a botched word (transposed,
  doubled, or missing internal letters) rather than a recognised
  shorthand.

A difference that is **only** about padding whitespace, case, or a
single trivial punctuation mark is CLEAN. Word-boundary spacing defects
that change tokenisation are NOT covered here — those are `spacing`
typos per sec. 1.

**Truncation-shaped values are CLEAN.** A value that ends mid-word, ends
on an unexpected character, or looks like a prefix of a longer canonical
form is CLEAN — even when surrounding values are full length. Truncation
is not verifiable from the value alone: it may be a deliberate
abbreviation, a redaction, or a fixed-width export limit. When many
distinct values cluster near the same maximum length and end mid-word,
the column was exported through a fixed-width field; treat all such
values as CLEAN.

---

## 3. Proper nouns and entities — character-level slips are OUT OF SCOPE

The default for any value that looks like a named entity is **CLEAN**,
and for a character-level slip that default is also the final answer. A
"proper noun" here covers personal names, geographic names, street
addresses, brand / company / product / organisation names, and
scientific binomials.

**The reason is verifiability, not rarity.** A column rarely carries
enough evidence to PROVE that a name token is wrong rather than a
rare-but-correct variant — a regional spelling, a legal-entity name, a
transliteration variant, or a deliberate stylisation. If the defect
cannot be verified from the value alone, it is NOT a typo for our
purposes, even when an outside reader believes they recognise the
intended canonical name. When in doubt about whether a token is a proper
noun, treat it as one and leave it CLEAN.

**Judge the TOKEN being corrected, not the whole value.** Only the
proper-noun token itself is protected. A misspelled COMMON word inside
an otherwise proper-noun value is still in scope: a value naming an
institution that contains an ordinary misspelled dictionary word (e.g. a
generic noun like `Hopsital` → `Hospital`, or a generic descriptor like
`Mian` → `Main`) is flag-able on that common-word token, while the name
token beside it is not.

**Structural defects on proper-noun values stay in scope** because they
are verifiable from the value alone without world knowledge:
- adjacent duplicated word (`Foo Bar Bar Baz` → `Foo Bar Baz`);
- obvious doubled or stray punctuation that breaks the value's shape
  without changing a word's letters.

Do NOT flag a name merely because it looks unusual, could plausibly be a
misspelling of a famous name, differs from your expected spelling, uses
non-English letters or diacritics, or appears only once in the column.

---

## 4. Alphanumeric codes / IDs — EXTRA CAUTION

Values that mix letters and digits within one token are codes; the
default is **CLEAN**. Flag only when the defect is a character-level slip
in the alphabetic portion that yields an obviously intended English word.
NEVER flag differences in spacing, casing, hyphenation, or punctuation
between segments of a code.

---

## 5. Output

Return a single JSON object — no Markdown fences, no surrounding text.

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

Return an empty list when nothing in the column is confidently a typo.
