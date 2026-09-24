from types import SimpleNamespace

from mepd.neb import NEB


def _neb(**params):
    defaults = dict(ts_converged_stop=True, ts_grad_thre=0.005, ts_spring_thre=0.005, barrier_thre=0.1)
    return SimpleNamespace(parameters=SimpleNamespace(**{**defaults, **params}))



class _Chain(list):
    def __init__(self, n=7, spring=0.001, ea=10.0):
        super().__init__(range(n))
        self.ts_triplet_gspring_infnorm = spring
        self._ea = ea

    def get_eA_chain(self):
        return self._ea


def test_ts_region_converged_ignores_the_rest_of_the_band():
    check = NEB._ts_region_converged
    assert check(_neb(), _Chain(), _Chain(), ind_ts=3, ts_grad=0.004)
    assert not check(_neb(), _Chain(), _Chain(), ind_ts=3, ts_grad=0.006)  # TS gradient
    assert not check(_neb(), _Chain(), _Chain(spring=0.01), ind_ts=3, ts_grad=0.004)  # springs at TS
    assert not check(_neb(), _Chain(ea=10.0), _Chain(ea=10.5), ind_ts=3, ts_grad=0.004)  # barrier moving
    assert not check(_neb(), _Chain(), _Chain(), ind_ts=6, ts_grad=0.0)  # TS guess at an endpoint
    assert not check(_neb(ts_converged_stop=False), _Chain(), _Chain(), ind_ts=3, ts_grad=0.0)
