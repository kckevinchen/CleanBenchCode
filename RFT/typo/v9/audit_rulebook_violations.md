# Benchmark audit: new rulebook violations

Benchmark: `benchmark_typo_20260512T213238Z_gpt5-family.jsonl`  
Total gold typo cells: **233**  

Two new rules in `RFT/typo_v9/prompts/rulebook.md`:

1. **Proper nouns are not verifiable** -- character-level slips on personal
   names, geographic names, brands, scientific binomials are CLEAN even if
   the column header says they're a name.
2. **Cell-level truncation is CLEAN** -- a single value that ends mid-word
   or is a clean prefix of a longer canonical form is CLEAN even when the
   surrounding column has full-length values.

Both rules are about *non-verifiability from the value alone*.

Trivial-correction (whitespace) cells are excluded from this audit -- they
are already excluded from official scoring per RFT.md sec.1.

---

## A. Truncation-rule violations -- **5 cells**

All cases below have `corrected_value` extending `original_value` -- the
canonical 'cell-level truncation' shape. Per the new rule, every one of
these should flip from `error` -> `clean` (or `ambiguous` if you want to
be conservative).

| Source | Row | Column | Original | Corrected |
|---|---|---|---|---|
| `3319d627.csv` | 61 | `Provider Name` | `NORTH ADAMS COMMONS NURSING & REHABILITATION CENTE` | `NORTH ADAMS COMMONS NURSING & REHABILITATION CENTER` |
| `80ee0112.csv` | 24 | `Desc` | `THEFT OF MOTOR VEHICLE PARTS OR ACCESSORIE` | `THEFT OF MOTOR VEHICLE PARTS OR ACCESSORIES` |
| `a421d963.csv` | 96 | `Division` | `Hospitality Fee Gaillard Cente` | `Hospitality Fee Gaillard Center` |
| `a66498cc.csv` | 95 | `District Name` | `Community Charter School Of Paterso` | `Community Charter School Of Paterson` |
| `c6e924fa.csv` | 57 | `Location` | `Tuggeranon` | `Tuggeranong` |

## B. Proper-noun-rule violations (strong) -- **35 cells**

In each row below, the differing token between original and corrected is
**not** a dictionary common word -- so the typo lands on the proper-noun
token itself (a surname, place name, scientific binomial, brand). Per the
new rule, these should flip from `error` -> `clean` (or `ambiguous`).

| Source | Row | Column | Original | Corrected | Differing tokens |
|---|---|---|---|---|---|
| `02973053.csv` | 7 | `Applicant Name` | `Bradford  Forfiet` | `Bradford  Forfeit` | ['Forfiet'] -> ['Forfeit'] |
| `02973053.csv` | 26 | `Applicant Name` | `Leicester (Forfiet)` | `Leicester (Forfeit)` | ['Forfiet'] -> ['Forfeit'] |
| `02973053.csv` | 58 | `Applicant Name` | `Ripton (Forfiet)` | `Ripton (Forfeit)` | ['Forfiet'] -> ['Forfeit'] |
| `02973053.csv` | 19 | `Applicant Name` | `Flecther` | `Fletcher` | ['Flecther'] -> ['Fletcher'] |
| `02973053.csv` | 43 | `Applicant Name` | `Old Cttage Lane Inc.` | `Old Cottage Lane Inc.` | ['Cttage'] -> ['Cottage'] |
| `3723eb12.csv` | 29 | `COUNTY NAME` | `Du Page` | `DuPage` | ['Du', 'Page'] -> ['DuPage'] |
| `3723eb12.csv` | 90 | `COUNTY NAME` | `Mc Lennan` | `McLennan` | ['Mc', 'Lennan'] -> ['McLennan'] |
| `4b553bb5.csv` | 79 | `Payee` | `94314 Mid Peninsula Regional Open Space District` | `94314 Midpeninsula Regional Open Space District` | ['Mid', 'Peninsula'] -> ['Midpeninsula'] |
| `4da01310.csv` | 90 | `Library` | `University of North Dakota - Thromodsgard Law Library` | `University of North Dakota - Thormodsgard Law Library` | ['Thromodsgard'] -> ['Thormodsgard'] |
| `6afc6078.csv` | 2 | `Agency Name` | `CHEATAM COUNTY HEALTH DEPARTMENT` | `CHEATHAM COUNTY HEALTH DEPARTMENT` | ['CHEATAM'] -> ['CHEATHAM'] |
| `6afc6078.csv` | 98 | `Agency Name` | `WILSON COUNTY CLERK - JIM GGODALL` | `WILSON COUNTY CLERK - JIM GOODALL` | ['GGODALL'] -> ['GOODALL'] |
| `8b48c63b.csv` | 18 | `Product` | `Pickalilly` | `Piccalilli` | ['Pickalilly'] -> ['Piccalilli'] |
| `8b48c63b.csv` | 22 | `Product` | `Vegimite` | `Vegemite` | ['Vegimite'] -> ['Vegemite'] |
| `8b48c63b.csv` | 77 | `Product` | `Brocollini` | `Broccolini` | ['Brocollini'] -> ['Broccolini'] |
| `8b48c63b.csv` | 79 | `Product` | `Brocollini` | `Broccolini` | ['Brocollini'] -> ['Broccolini'] |
| `8b48c63b.csv` | 88 | `Product` | `Yellow Fin Tuna` | `Yellowfin Tuna` | ['Yellow', 'Fin'] -> ['Yellowfin'] |
| `8b48c63b.csv` | 91 | `Product` | `Yellow Fin Tuna` | `Yellowfin Tuna` | ['Yellow', 'Fin'] -> ['Yellowfin'] |
| `8b48c63b.csv` | 98 | `Product` | `Scalops` | `Scallops` | ['Scalops'] -> ['Scallops'] |
| `9cac5c69.csv` | 27 | `Company` | `Seacost Commerce Bank` | `Seacoast Commerce Bank` | ['Seacost'] -> ['Seacoast'] |
| `a9351b11.csv` | 37 | `Albright Middle` | `O'Donell Middle` | `O'Donnell Middle` | ["O'Donell"] -> ["O'Donnell"] |
| `addf5bdd.csv` | 19 | `Common animal name` | `Spragues's Pipits` | `Sprague's Pipits` | ["Spragues's"] -> ["Sprague's"] |
| `addf5bdd.csv` | 31 | `Nom commun d'animal` | `Apternotus leptrorhynchus` | `Apteronotus leptorhynchus` | ['Apternotus', 'leptrorhynchus'] -> ['Apteronotus', 'leptorhynchus'] |
| `addf5bdd.csv` | 40 | `Nom commun d'animal` | `loup à ocelles (Anarhichthys ocellatus)` | `loup à ocelles (Anarrhichthys ocellatus)` | ['Anarhichthys'] -> ['Anarrhichthys'] |
| `addf5bdd.csv` | 49 | `Nom commun d'animal` | `laquaiches argentée` | `laquaiche argentée` | ['laquaiches'] -> ['laquaiche'] |
| `addf5bdd.csv` | 68 | `Nom commun d'animal` | `requin, gris de récif (Carcharhinus amblyrohynchos)` | `requin, gris de récif (Carcharhinus amblyrhynchos)` | ['amblyrohynchos'] -> ['amblyrhynchos'] |
| `addf5bdd.csv` | 72 | `Nom commun d'animal` | `truite arc-en-ciel (Oncorhynchuk mykiss)` | `truite arc-en-ciel (Oncorhynchus mykiss)` | ['Oncorhynchuk'] -> ['Oncorhynchus'] |
| `addf5bdd.csv` | 97 | `Common animal name` | `Snakes, Eastern Garter (Thamnophis s. surtalis)` | `Snakes, Eastern Garter (Thamnophis s. sirtalis)` | ['surtalis'] -> ['sirtalis'] |
| `aec55fe9.csv` | 5 | `City` | `Stockholme` | `Stockholm` | ['Stockholme'] -> ['Stockholm'] |
| `ba96b229.csv` | 41 | `Supplier Name` | `Iliffee Media Limited` | `Iliffe Media Limited` | ['Iliffee'] -> ['Iliffe'] |
| `ba96b229.csv` | 42 | `Supplier Name` | `Iliffee Media Limited` | `Iliffe Media Limited` | ['Iliffee'] -> ['Iliffe'] |
| `c181da95.csv` | 24 | `Site Name` | `VILLAREAL EL` | `VILLARREAL EL` | ['VILLAREAL'] -> ['VILLARREAL'] |
| `d7901995.csv` | 54 | `Community` | `Lac des Milles Lacs` | `Lac des Mille Lacs` | ['Milles'] -> ['Mille'] |
| `de1df911.csv` | 9 | `City` | `Brenalillo` | `Bernalillo` | ['Brenalillo'] -> ['Bernalillo'] |
| `de1df911.csv` | 15 | `City` | `Cimarrons` | `Cimarron` | ['Cimarrons'] -> ['Cimarron'] |
| `e612e69f.csv` | 55 | `Entity Name` | `23050-MC COMB LSD` | `23050-MCCOMB LSD` | ['MC', 'COMB'] -> ['MCCOMB'] |

## C. Borderline (common-word slip inside a name) -- **14 cells**

The slipped token resolves to a common dictionary word (e.g.
`INSURANGE -> INSURANCE`, `LIASON -> LIAISON`, `Sliky -> Silky`). Per the
current rulebook these are **IN scope** -- the typo is on a common-word
token even though the surrounding context is a proper name. No change
needed unless you want to tighten further.

| Source | Row | Column | Original | Corrected | Differing tokens |
|---|---|---|---|---|---|
| `02973053.csv` | 75 | `Applicant Name` | `West Shore Asssociation` | `West Shore Association` | ['Asssociation'] -> ['Association'] |
| `4b553bb5.csv` | 58 | `Payee` | `93633 Inland Valley Developmental Agency` | `93633 Inland Valley Development Agency` | ['Developmental'] -> ['Development'] |
| `531c1dd0.csv` | 11 | `SUPPLIER` | `GRAHAM INDEPENDANT` | `GRAHAM INDEPENDENT` | ['INDEPENDANT'] -> ['INDEPENDENT'] |
| `531c1dd0.csv` | 72 | `SUPPLIER` | `GRAHAM INDEPENDANT` | `GRAHAM INDEPENDENT` | ['INDEPENDANT'] -> ['INDEPENDENT'] |
| `63408b83.csv` | 25 | `VENDOR NAME` | `AQUATIC ENVIROMENTAL` | `AQUATIC ENVIRONMENTAL` | ['ENVIROMENTAL'] -> ['ENVIRONMENTAL'] |
| `6afc6078.csv` | 15 | `Agency Name` | `DCS - COURT LIASON JUVENILE COURT B` | `DCS - COURT LIAISON JUVENILE COURT B` | ['LIASON'] -> ['LIAISON'] |
| `6afc6078.csv` | 57 | `Agency Name` | `WOODLAND HILLS YOUTH DEVELOPMETN CT` | `WOODLAND HILLS YOUTH DEVELOPMENT CT` | ['DEVELOPMETN'] -> ['DEVELOPMENT'] |
| `6afc6078.csv` | 96 | `Agency Name` | `Wilson Country Drug Court` | `Wilson County Drug Court` | ['Country'] -> ['County'] |
| `76b2ac82.csv` | 69 | `school` | `Sibley-Ocheyedan Community SchoolDistrict` | `Sibley-Ocheyedan Community School District` | ['SchoolDistrict'] -> ['School', 'District'] |
| `967ab301.csv` | 55 | `City` | `Pilot Mountatin` | `Pilot Mountain` | ['Mountatin'] -> ['Mountain'] |
| `c6e924fa.csv` | 22 | `Location` | `Toowoomba Nour` | `Toowoomba North` | ['Nour'] -> ['North'] |
| `dc5312fb.csv` | 28 | `Name` | `RYAN INSURANGE GROUP` | `RYAN INSURANCE GROUP` | ['INSURANGE'] -> ['INSURANCE'] |
| `e9e32f53.csv` | 22 | `CCG / Health Board / NHS Trust / GP` | `Dr Sood and partenrs - The Family Medical Centre` | `Dr Sood and partners - The Family Medical Centre` | ['partenrs'] -> ['partners'] |
| `fc844b89.csv` | 19 | `Product` | `Smooth Sliky Salty` | `Smooth Silky Salty` | ['Sliky'] -> ['Silky'] |

## D. Structural defects on proper-noun values -- 9 cells

Adjacent duplicated word, doubled punctuation, or spaced-initialism
collapse (e.g. `A T & T -> AT&T`, `Du Page -> DuPage`, `MC COMB -> MCCOMB`).
These remain **IN scope** per current rulebook sec.3 because structural
defects are verifiable from the value alone. No change needed.

| Source | Row | Column | Original | Corrected |
|---|---|---|---|---|
| `0e88fcab.csv` | 50 | `Cost Centre Name` | `M R I  / C T` | `MRI / CT` |
| `0e88fcab.csv` | 46 | `Supplier Name` | `I D MEDICAL` | `ID MEDICAL` |
| `41d92bb8.csv` | 53 | `VENDOR NAME` | `A T & T` | `AT&T` |
| `dc5312fb.csv` | 15 | `Name` | `COUNTY HALL (ESSEX COUNCIL COUNCIL)` | `COUNTY HALL (ESSEX COUNCIL)` |
| `e612e69f.csv` | 48 | `IRN Organization Name` | `HURON CITY CITY SD` | `HURON CITY SD` |
| `e612e69f.csv` | 55 | `IRN Organization Name` | `MCCOMB LOCAL LOCAL SD` | `MCCOMB LOCAL SD` |
| `e612e69f.csv` | 69 | `IRN Organization Name` | `BUCKEYE LOCAL LOCAL SD` | `BUCKEYE LOCAL SD` |
| `e612e69f.csv` | 91 | `IRN Organization Name` | `INDIAN VALLEY LOCAL LOCAL SD` | `INDIAN VALLEY LOCAL SD` |
| `f1dc3e29.csv` | 47 | `VENDOR NAME` | `A T & T` | `AT&T` |

---

## Recommended next step

Re-label the **5 truncation cells** and **35 strong
proper-noun cells** in the benchmark from `error` -> `clean` (or
`ambiguous`), then rerun `RFT/typo_v9/scripts/build_v9_splits.py` to
propagate the new gold + the new rulebook into the v9 training data.

Per project rule D10 (don't tune against the benchmark), no changes were
made automatically -- this report is for your review only.