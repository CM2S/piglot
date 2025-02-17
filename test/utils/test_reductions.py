import pytest
from piglot.utils.reductions import AVAILABLE_REDUCTIONS, Reduction, NegateReduction


@pytest.mark.parametrize('reduction', AVAILABLE_REDUCTIONS.values())
def test_simple_reductions(reduction: Reduction):
    reduction.test_reduction()


@pytest.mark.parametrize('reduction', AVAILABLE_REDUCTIONS.values())
def test_negate_simple_reductions(reduction: Reduction):
    negate_reduction = NegateReduction(reduction)
    negate_reduction.test_reduction()
