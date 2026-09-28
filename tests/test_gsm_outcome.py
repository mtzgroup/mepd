"""GSM._gsm_outcome: how a molecularGSM run ended, from its stdout."""
from mepd.pathminimizers.gsm import GSM

MAX_ITER = """ oi: 300 nmax: 3 TSnode0: 4 overlapn: 1
 opt_iters over: totalgrad: 0.147 gradrms: 0.0038 tgrads: 3562  ol(1): 0.56 max E:  93.4 Erxn: -43.3 nmax:  3 TSnode:  4    -max_iter-
"""
XTS = """ oi: 17 nmax: 4 TSnode0: 4 overlapn: 1
 opt_iters over: totalgrad: 0.052 gradrms: 0.0019 tgrads: 812  ol(1): 0.02 max E:  32.8 Erxn: -21.7 nmax:  4 TSnode:  4    -XTS-
"""
CRASHED = """ Opt step:  3  gqc: 0.020 ss: 0.182 (DMAX: 0.185) predE: -2.27  E(M): 27.49 gRMS: 0.0075
"""


def outcome(text, **kw):
    return GSM._gsm_outcome(text, **{"seeded": True, "early_stopped": False, "n_calls": 10, **kw})


def test_out_of_iterations_is_not_converged():
    o = outcome(MAX_ITER)
    assert (o["stop"], o["converged"], o["opt_iters"], o["gradrms"]) == ("max_iter", False, 300, 0.0038)


def test_the_xts_marker_means_converged_even_above_conv_tol():
    o = outcome(XTS)   # XTS runs end at gradrms ~2e-3, above conv_tol 5e-4
    assert (o["stop"], o["converged"], o["opt_iters"]) == ("XTS", True, 17)


def test_a_run_without_a_closing_line_is_unknown():
    o = outcome(CRASHED)
    assert (o["stop"], o["converged"]) == ("unknown", False)


def test_an_early_stop_is_never_converged():
    o = outcome(XTS, early_stopped=True)
    assert (o["stop"], o["converged"]) == ("early_stop", False)
