#!/usr/bin/env python3
"""Run upstream HoloClean hospital example and export its repaired dataframe."""
from __future__ import print_function
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "holoclean_repo"
sys.path.insert(0, str(UPSTREAM))

import holoclean  # noqa: E402
from detect import NullDetector, ViolationDetector  # noqa: E402
from repair.featurize import (  # noqa: E402
    ConstraintFeaturizer,
    FreqFeaturizer,
    InitAttrFeaturizer,
    OccurAttrFeaturizer,
)

hc = holoclean.HoloClean(
    db_name="holo",
    domain_thresh_1=0,
    domain_thresh_2=0,
    weak_label_thresh=0.99,
    max_domain=10000,
    cor_strength=0.6,
    nb_cor_strength=0.8,
    epochs=10,
    weight_decay=0.01,
    learning_rate=0.001,
    threads=1,
    batch_size=1,
    verbose=True,
    timeout=3 * 60000,
    feature_norm=False,
    weight_norm=False,
    print_fw=True,
).session
hc.load_data("hospital", str(UPSTREAM / "testdata" / "hospital.csv"))
hc.load_dcs(str(UPSTREAM / "testdata" / "hospital_constraints.txt"))
hc.ds.set_constraints(hc.get_dcs())
hc.detect_errors([NullDetector(), ViolationDetector()])
hc.setup_domain()
hc.repair_errors([InitAttrFeaturizer(), OccurAttrFeaturizer(), FreqFeaturizer(), ConstraintFeaturizer()])
out = ROOT / "example_hospital_repaired.csv"
hc.ds.repaired_data.df.to_csv(str(out), index=False)
print("wrote", out, hc.ds.repaired_data.df.shape)
