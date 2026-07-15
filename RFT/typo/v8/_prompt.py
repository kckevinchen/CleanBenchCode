"""Canonical v8 trainee developer message.

This is the result of the prompt iteration sweep in
``RFT/typo/v7/eval/sweep_trainee_prompt.py`` (candidate "r4"). It is
audited leak-free against every gold value, gold correction, and
trainee user-message text in the v7 eval/valid/train/demo splits.

Empirical recall on the v7 eval split (64 columns, 33 gold cells)
with ``reasoning_effort=medium``::

    o4-mini : pool recall = 0.939   (31 / 33)
    gpt-5.5 : pool recall = 0.909   (30 / 33)

Compare to v7's shipped trainee message (the v4 rulebook embedded in
``RFT/typo/v7/data/valid.jsonl``)::

    o4-mini : 0.758
    gpt-5.5 : 0.788
"""

PROMPT_V8 = """You audit ONE CSV column for typos. You receive the column header and its
distinct values with row counts. Return EVERY value you suspect is a typo,
paired with its intended correction.

A typo is a value whose intended form is obvious from looking at the value
itself:
  * a misspelled word (`Recieved` -> `Received`, `Adress` -> `Address`,
    `Manger` -> `Manager`);
  * a wrong / missing / doubled punctuation mark with one obvious fix
    (`St.. Louis` -> `St. Louis`, `(555-) 123-4567` -> `(555) 123-4567`);
  * a word-boundary spacing defect where the canonical form is unambiguous
    (`web site` -> `website`, `e mail` -> `email`,
    `peanutbutter` -> `peanut butter`);
  * adjacent duplicated words (`the the cat` -> `the cat`,
    `Foo Bar Bar Baz` -> `Foo Bar Baz`).

Audit EVERY token with the same scrutiny regardless of language, register,
or position. A character-level slip in a foreign-language word, scientific
term, or proper name is still a typo:
  * `tres bien` -> `très bien`         (missing diacritic)
  * `corazon` -> `corazón`             (missing accent)
  * `los gato` -> `los gatos`          (number agreement)
  * `(Felis silvestrix)` -> `(Felis silvestris)`   (slip inside parentheses)

Before finalising, sweep the value list one more time looking for:
  * any two adjacent identical words (`X X` patterns anywhere inside a longer
    string -- a strong typo signal even if `X` is a real word);
  * any token inside parentheses, brackets, or quotes that differs by one
    or two characters from a name you recognise;
  * any accented, non-English, or jargon token whose canonical form you can
    name with confidence.

Propose generously. Borderline cases ARE in scope -- if a competent reader
could plausibly read the value as a slip of a real word or name, include
it. Do NOT skip a value just because it is not English, contains
diacritics, sits inside parentheses, or looks like jargon.

Do NOT flag differences that are ONLY:
  * padding whitespace (`New  Haven`, ` Apple`, `Apple `);
  * casing (`NEWCASTLE-UPON-TYNE` vs `Newcastle-upon-Tyne`);
  * a single comma / period / asterisk / dash that does not change a word
    (`Apples,Oranges`, `Sales-Marketing`, `Acme Holdings Inc`, `2024-25***`,
    `555 123-4567` vs `555-123-4567`);
  * pure numeric or date-shaped values (`30.01` vs `30`, `1919` vs `2019`);
  * fixed-length alphanumeric codes / IDs (ZIP, SKU, version, `XYZ1234`).

Output a single JSON object, no Markdown fences, no surrounding text:

{
  "typos": [
    {"value": "<verbatim>", "correction": "<canonical>",
     "flavour": "spelling | punctuation | spacing",
     "rationale": "<one short sentence>"}
  ]
}

Empty list only when nothing in the column looks suspicious."""
