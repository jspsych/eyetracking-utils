__version__ = "0.3.0"

# Import custom layers so they get registered at import time
from et_util.custom_layers import (
    SimpleTimeDistributed,
    MaskedWeightedRidgeRegressionLayer,
    MaskInspectorLayer,
)

# Import custom loss so it gets registered at import time
from et_util.custom_loss import normalized_weighted_euc_dist
