"""v10 trainee developer message (recall-push wording REMOVED, 2026-06-01).

v10 forks v9's trainee prompt with ONE change: the two explicit
"recall-push" passages are removed, because v10 does not chase the
*trivial* typo cells those passages were buying back.

Removed vs ``PROMPT_V9``:
  1. The "Your job is RECALL: a separate precision check catches wrong
     guesses, so propose generously and do not self-censor borderline
     ordinary-word cases" sentence in the opening paragraph.
  2. The "Borderline cases ARE in scope: if a competent reader could
     read the value as a slip of a real word, include it" clause in the
     foreign-language paragraph.

Everything else (typo definition + examples, the any-language /
adjacent-duplicate sweep, proper-noun out-of-scope rules, the
do-not-flag list incl. diacritics, the JSON output schema) is identical
to ``PROMPT_V9``.

Recall measured on the RFT base o4-mini (67 gold-typo cols,
reasoning_effort=low, ``RFT.typo.v10.eval.recall_bench``):

    metric        PROMPT_V9 (v9)    PROMPT_V10 (this)
    pool          181/201 = 0.9005  175/201 = 0.8706
    non-trivial    70/80  = 0.8750   71/80  = 0.8875
    trivial       111/121 = 0.9174  104/121 = 0.8595

i.e. the wording removal costs ~6 pool cells, ALL of them in the
*trivial* bucket (non-trivial recall is flat / +1, within o4-mini's
run-to-run noise). v10 accepts the trivial-recall loss by design.

v10 reuses v9's precision grader and rulebook unchanged
(``RFT.typo.v9._grader_variants``, ``RFT/typo/v9/prompts/rulebook.md``);
only the trainee prompt differs. The data splits
(``RFT/typo/v10/data/*.jsonl``) are v9's splits with ``messages[0].content``
retargeted to ``PROMPT_V10`` (gold cells byte-identical, eval still
disjoint); see ``RFT/typo/v10/scripts/build_v10_splits.py``.
"""

PROMPT_V10 = """You audit ONE CSV column for typos. You get the header and the column's
distinct values with counts. Return every value you suspect is a typo with its
intended correction.

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
gatos`). Do not skip a value for being non-English or accented. Also flag any
two adjacent identical words (`X X`) anywhere in a string, even a name.

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
