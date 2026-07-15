"""v9 trainee developer message (SHORTENED + diacritics descoped, 2026-05-28).

Two changes from the original 973-token v9 prompt:
  1. Prose compression (the "candA" rewrite): cut redundancy and trimmed the
     proper-noun / do-not-flag sections, while preserving every recall driver
     (typo definition + examples, "propose generously", the adjacent-duplicate
     sweep).
  2. Diacritics put OUT OF SCOPE -- an explicit do-not-flag clause ("accent
     presence/absence is never a typo on its own"), a deliberate precision call.

Now 734 tokens (-25%, 76 -> 48 lines). Recall verified on the RFT base o4-mini
(67 gold-typo cols, reasoning_effort=low, 3 stochastic samples -- o4-mini is a
reasoning model so recall varies run to run): pool 196-200 vs the original's
201-205, non-trivial 87-90 vs 90-94, trivial 107-111 vs 110-114 -- i.e. a small
(~3-5 cell) recall reduction, accepted in exchange for the diacritic rule.
The pure-compression "candA" variant kept recall equal-or-better at 696 tokens
and is preserved at outputs/v9_4omini_fullbench/candidates/v9_short_a.txt if you
want that recall back.

The data splits (RFT/typo/v9/data/*.jsonl) WERE rebuilt to bake this prompt
(build_v9_splits.py --apply-report: prompt retargeted, gold reconciliation
byte-identical, eval still disjoint). Backups: _prompt.py.pre_shorten.bak
(original 973-token prompt), data.pre_shorten.bak (pre-rebuild splits).

v9 design (unchanged): the TRAINEE optimises RECALL (surface every plausibly-
suspicious ordinary-word value; proper nouns out of scope for character-level
slips); the GRADER optimises PRECISION. See RFT/typo/v9/prompts/rulebook.md and
RFT/typo/v9/_grader_variants.py.
"""

PROMPT_V9 = """You audit ONE CSV column for typos. You get the header and the column's
distinct values with counts. Return EVERY value you suspect is a typo with its
intended correction. Your job is RECALL: a separate precision check catches
wrong guesses, so propose generously and do not self-censor borderline
ordinary-word cases.

A typo is a value whose intended form is obvious from the value alone:
  * misspelling (`Recieved` -> `Received`, `Adress` -> `Address`,
    `Seperate` -> `Separate`);
  * wrong / missing / doubled punctuation with one obvious fix
    (`St.. Louis` -> `St. Louis`, `(555-) 123-4567` -> `(555) 123-4567`);
  * a word-boundary spacing slip with an unambiguous canonical form
    (`web site` -> `website`, `e mail` -> `email`,
    `peanutbutter` -> `peanut butter`);
  * adjacent duplicated words (`the the cat` -> `the cat`).

Audit every ordinary-word token with equal scrutiny in ANY language -- a
character slip in a common foreign word is still a typo (`los gato` -> `los
gatos`). Borderline cases ARE in scope: if a competent reader could read the
value as a slip of a real word, include it; do not skip a value for being
non-English or accented. Also flag any two adjacent identical words (`X X`)
anywhere in a string, even a name.

Names, brands, places, addresses, and scientific names are OUT OF SCOPE for
character-level fixes: never "correct" a proper-noun token (`Cincinatti`,
`Microsft`, `Felis silvestrix`) -- you cannot tell a mistyped name from a rare
real one. Two exceptions stay in scope, because the defect is visible from the
string alone:
  * structural defects -- adjacent duplicated words, doubled / stray
    punctuation (`St.. Louis` -> `St. Louis`);
  * a COMMON-WORD token inside a name carrying a clear slip -- fix only that
    token (`Bright Hosptial Trust` -> `Bright Hospital Trust`).
Single-letter tokens spaced apart are a spacing defect even in a name -- close
them up, keeping any `&` / punctuation (`I B M` -> `IBM`, `U S A` -> `USA`).

Do NOT flag differences that are ONLY: padding whitespace (`New  Haven`,
`Apple `); casing; a single comma / period / asterisk / dash that does not
change a word (`Apples,Oranges`, `Sales-Marketing`, `2024-25***`); pure numeric
or date-shaped values (`12.50` vs `12`, `1919` vs `2019`); fixed-length
codes / IDs (ZIP, SKU, version, `XYZ1234`); or accent marks / diacritics
(`corazon` vs `corazón`, `tres bien` vs `très bien`, `Felis silvestris` vs
`Felis silvéstris`) -- accent presence/absence is never a typo on its own.

Output a single JSON object, no Markdown fences, no surrounding text:
{"typos": [{"value": "<verbatim>", "correction": "<canonical>",
  "flavour": "spelling | punctuation | spacing",
  "rationale": "<one short sentence>"}]}
Empty list only when nothing in the column looks suspicious."""
