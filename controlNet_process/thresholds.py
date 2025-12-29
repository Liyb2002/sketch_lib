# graph_building/thresholds.py
"""
Centralized thresholds for graph relation construction.

These values are intentionally simple and global.
Changing them alters graph topology.
"""

# ---------------------------------------------------------------------
# SAME-PAIR RELATION THRESHOLDS
# ---------------------------------------------------------------------

# Minimum confidence required to keep a same_pair relation.
# Current behavior: accept all same_pairs.
same_pair_relation_threshold_confidence = 0.0


# ---------------------------------------------------------------------
# CONNECT (GEOMETRIC) RELATION THRESHOLDS
# ---------------------------------------------------------------------

# Relative tolerance w.r.t. median component extent
connect_relation_threshold_ratio = 0.02

# Absolute tolerance in world units
connect_relation_threshold_abs = 0.0


# ---------------------------------------------------------------------
# NUMERICAL STABILITY (DO NOT CHANGE UNLESS YOU KNOW WHY)
# ---------------------------------------------------------------------

floating_point_eps = 1e-12
