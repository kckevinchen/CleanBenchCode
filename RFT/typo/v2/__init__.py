"""Typo RFT v2 bundle (binary task, no trivial errors).

See README.md for the full description. Key differences from v1:

* Binary output: model emits ``typo`` or ``clean`` only. No ``ambiguous``.
* No trivial-edit training data: gold typos that are pure whitespace /
  casing / punctuation rewrites are filtered out (147 / 243 rows in
  the source benchmark).
* Ambiguous-gold eval rows are credited on either binary answer.
"""
