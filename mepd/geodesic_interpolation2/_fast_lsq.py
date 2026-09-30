"""
_fast_lsq.py: a drop-in, faster ``least_squares`` for the geodesic code.

The geodesic smoother and the midpoint finder both call
``scipy.optimize.least_squares(method='trf', tr_solver='lsmr')`` with a
callable sparse Jacobian and no bounds.  For the problem sizes we see
(a few hundred to a few thousand residuals) almost all of that time is
Python overhead in ``scipy.sparse.linalg.lsmr``: every one of its ~100
iterations per trust-region step goes through several LinearOperator
wrappers.

This module re-implements exactly that code path -- SciPy's
``trf_no_bounds`` with its ``lsmr`` branch, following SciPy 1.15-1.18 line
by line -- and runs the LSMR iterations in a numba-compiled kernel that
performs the same floating-point operations in the same order (same
sparse mat-vec loops as ``_sparsetools``, BLAS ``ddot`` for the norms, same
Givens rotations).  Everything outside the LSMR inner loop reuses SciPy's
own helpers.  The results are therefore the same as SciPy's to round-off
(in practice bit-for-bit on the benchmark set).

Only the options the geodesic code uses are supported; anything else, or
a missing numba, falls back to ``scipy.optimize.least_squares``.
"""

from math import sqrt

import numpy as np
from numpy.linalg import norm
import scipy.sparse as sps
from scipy.optimize import OptimizeResult, least_squares as _scipy_least_squares

try:  # SciPy private helpers (stable across 1.x); fall back if they move.
    from scipy.linalg import qr
    from scipy.optimize._lsq.common import (
        solve_trust_region_2d, minimize_quadratic_1d, build_quadratic_1d,
        evaluate_quadratic, compute_grad, compute_jac_scale, check_termination,
        update_tr_radius, scale_for_robust_loss_function)
    from scipy.optimize._lsq.least_squares import (
        construct_loss_function, check_tolerance)
    _HAVE_SCIPY_INTERNALS = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY_INTERNALS = False

try:
    import numba as _nb
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


ENABLED = _HAVE_NUMBA and _HAVE_SCIPY_INTERNALS


if _HAVE_NUMBA:

    @_nb.njit(cache=True, inline="always")
    def _sign(a):
        if a > 0.0:
            return 1.0
        elif a < 0.0:
            return -1.0
        return 0.0 * a  # +-0.0 (and nan) like np.sign

    @_nb.njit(cache=True)
    def _sym_ortho(a, b):
        # scipy.sparse.linalg._isolve.lsqr._sym_ortho
        if b == 0.0:
            return _sign(a), 0.0, abs(a)
        elif a == 0.0:
            return 0.0, _sign(b), abs(b)
        elif abs(b) > abs(a):
            tau = a / b
            s = _sign(b) / sqrt(1.0 + tau * tau)
            c = s * tau
            r = b / s
        else:
            tau = b / a
            c = _sign(a) / sqrt(1.0 + tau * tau)
            s = c * tau
            r = a / c
        return c, s, r

    @_nb.njit(cache=True)
    def _csr_matvec(m, indptr, indices, data, x, y):
        # _sparsetools csr_matvec: y[i] = sum_jj data[jj] * x[indices[jj]]
        for i in range(m):
            s = 0.0
            for jj in range(indptr[i], indptr[i + 1]):
                s += data[jj] * x[indices[jj]]
            y[i] = s

    @_nb.njit(cache=True)
    def _csrT_matvec(m, indptr, indices, data, u, y):
        # (J.T).dot(u) for csr J == _sparsetools csc_matvec on J.T
        for k in range(y.shape[0]):
            y[k] = 0.0
        for i in range(m):
            ui = u[i]
            for jj in range(indptr[i], indptr[i + 1]):
                y[indices[jj]] += data[jj] * ui

    @_nb.njit(cache=True)
    def _csr_to_ell(m, indptr, indices, data):
        # Row-padded ("ELLPACK") copy of a CSR matrix: row i keeps its entries in
        # CSR order, followed by zeros.  A row sum over it performs the same
        # additions as csr_matvec plus trailing "+ 0.0" terms, which are exact.
        K = 0
        for i in range(m):
            K = max(K, indptr[i + 1] - indptr[i])
        D = np.zeros((m, K))
        I = np.zeros((m, K), dtype=np.int32)
        for i in range(m):
            t = 0
            for jj in range(indptr[i], indptr[i + 1]):
                D[i, t] = data[jj]
                I[i, t] = indices[jj]
                t += 1
        return D, I

    @_nb.njit(cache=True)
    def _csr_to_csc(m, n, indptr, indices, data):
        # Column lists in ascending row order (stable counting sort), so a column
        # sum adds the terms in the order csc_matvec's scatter does.
        cptr = np.zeros(n + 1, dtype=np.int64)
        for jj in range(indptr[m]):
            cptr[indices[jj] + 1] += 1
        for j in range(n):
            cptr[j + 1] += cptr[j]
        fill = cptr[:-1].copy()
        rows = np.empty(indptr[m], dtype=np.int32)
        cdata = np.empty(indptr[m])
        for i in range(m):
            for jj in range(indptr[i], indptr[i + 1]):
                j = indices[jj]
                rows[fill[j]] = i
                cdata[fill[j]] = data[jj]
                fill[j] += 1
        return cptr, rows, cdata

    @_nb.njit(cache=True)
    def _ell_matvec_update(D, I, x, u, factor):
        # u = u * factor + J @ x   (== "u *= factor; u += J.dot(x)")
        m, K = D.shape
        for i in range(m):
            s = 0.0
            for t in range(K):
                s += D[i, t] * x[I[i, t]]
            u[i] = u[i] * factor + s

    @_nb.njit(cache=True)
    def _csc_rmatvec(cptr, rows, cdata, u, y):
        # y = J.T @ u, summing each column in ascending row order
        for j in range(y.shape[0]):
            s = 0.0
            for jj in range(cptr[j], cptr[j + 1]):
                s += cdata[jj] * u[rows[jj]]
            y[j] = s

    @_nb.njit(cache=True)
    def _nrm(x):
        # numpy.linalg.norm(x) for a real 1-D array is sqrt(dot(x, x))
        return sqrt(np.dot(x, x))

    @_nb.njit(cache=True)
    def _lsmr_scaled(indptr, indices, data, m, n, dscale, b, damp,
                     atol, btol, conlim, maxiter):
        """scipy.sparse.linalg.lsmr on A = J @ diag(dscale) (x0=None)."""
        tmp_n = np.empty(n)
        Atu = np.empty(n)
        D, I = _csr_to_ell(m, indptr, indices, data)
        cptr, crows, cdata = _csr_to_csc(m, n, indptr, indices, data)

        u = b
        normb = _nrm(b)
        x = np.zeros(n)
        beta = normb
        if beta > 0:
            u = (1.0 / beta) * u
            _csc_rmatvec(cptr, crows, cdata, u, Atu)
            v = dscale * Atu
            alpha = _nrm(v)
        else:
            v = np.zeros(n)
            alpha = 0.0
        if alpha > 0:
            v = (1.0 / alpha) * v

        itn = 0
        zetabar = alpha * beta
        alphabar = alpha
        rho = 1.0
        rhobar = 1.0
        cbar = 1.0
        sbar = 0.0
        h = v.copy()
        hbar = np.zeros(n)
        betadd = beta
        betad = 0.0
        rhodold = 1.0
        tautildeold = 0.0
        thetatilde = 0.0
        zeta = 0.0
        d = 0.0
        normA2 = alpha * alpha
        maxrbar = 0.0
        minrbar = 1e+100
        normA = sqrt(normA2)
        condA = 1.0
        normx = 0.0
        istop = 0
        ctol = 0.0
        if conlim > 0:
            ctol = 1.0 / conlim
        normr = beta
        normar = alpha * beta
        if normar == 0:
            return x, itn
        if normb == 0:
            x[:] = 0.0
            return x, itn

        while itn < maxiter:
            itn = itn + 1
            # u = A v - alpha u
            for k in range(n):
                tmp_n[k] = v[k] * dscale[k]
            _ell_matvec_update(D, I, tmp_n, u, -alpha)
            beta = _nrm(u)
            if beta > 0:
                u *= (1.0 / beta)
                v *= -beta
                _csc_rmatvec(cptr, crows, cdata, u, Atu)
                for k in range(n):
                    v[k] += dscale[k] * Atu[k]
                alpha = _nrm(v)
                if alpha > 0:
                    v *= (1.0 / alpha)

            chat, shat, alphahat = _sym_ortho(alphabar, damp)
            rhoold = rho
            c, s, rho = _sym_ortho(alphahat, beta)
            thetanew = s * alpha
            alphabar = c * alpha
            rhobarold = rhobar
            zetaold = zeta
            thetabar = sbar * rho
            rhotemp = cbar * rho
            cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
            zeta = cbar * zetabar
            zetabar = - sbar * zetabar

            fh = - (thetabar * rho / (rhoold * rhobarold))
            fx = (zeta / (rho * rhobar))
            fhh = - (thetanew / rho)
            for k in range(n):
                hbar[k] *= fh
                hbar[k] += h[k]
                x[k] += fx * hbar[k]
                h[k] *= fhh
                h[k] += v[k]

            betaacute = chat * betadd
            betacheck = -shat * betadd
            betahat = c * betaacute
            betadd = -s * betaacute
            thetatildeold = thetatilde
            ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
            thetatilde = stildeold * rhobar
            rhodold = ctildeold * rhobar
            betad = - stildeold * betad + ctildeold * betahat
            tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
            taud = (zeta - thetatilde * tautildeold) / rhodold
            d = d + betacheck * betacheck
            normr = sqrt(d + (betad - taud) ** 2 + betadd * betadd)
            normA2 = normA2 + beta * beta
            normA = sqrt(normA2)
            normA2 = normA2 + alpha * alpha
            maxrbar = max(maxrbar, rhobarold)
            if itn > 1:
                minrbar = min(minrbar, rhobarold)
            condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)
            normar = abs(zetabar)
            normx = _nrm(x)
            test1 = normr / normb
            if (normA * normr) != 0:
                test2 = normar / (normA * normr)
            else:
                test2 = np.inf
            test3 = 1.0 / condA
            t1 = test1 / (1.0 + normA * normx / normb)
            rtol = btol + atol * normA * normx / normb
            if itn >= maxiter:
                istop = 7
            if 1.0 + test3 <= 1.0:
                istop = 6
            if 1.0 + test2 <= 1.0:
                istop = 5
            if 1.0 + t1 <= 1.0:
                istop = 4
            if test3 <= ctol:
                istop = 3
            if test2 <= atol:
                istop = 2
            if test1 <= rtol:
                istop = 1
            if istop > 0:
                break
        return x, itn


class _RightScaled:
    """J @ diag(d) with just the ``dot`` SciPy's helpers need (csr J)."""

    __slots__ = ("J", "d", "shape")

    def __init__(self, J, d):
        self.J, self.d, self.shape = J, d, J.shape

    def dot(self, x):
        if x.ndim == 1:
            return self.J.dot(x * self.d)
        return self.J.dot(x * self.d[:, np.newaxis])


def _trf_no_bounds_lsmr(fun, jac, x0, f0, J0, ftol, xtol, gtol, max_nfev,
                        x_scale, loss_function):
    """scipy.optimize._lsq.trf.trf_no_bounds, tr_solver='lsmr' branch only.

    Default tr_options (damp=0, regularize=True, lsmr atol=btol=1e-6,
    conlim=1e8, maxiter=min(m, n)).  Returns an OptimizeResult with only
    ``x``, ``status`` and ``nfev`` filled in.
    """
    x = x0.copy()
    f = f0
    nfev = 1
    J = J0
    m, n = J.shape

    if loss_function is not None:
        rho = loss_function(f)
        cost = 0.5 * np.sum(rho[0])
        J, f = scale_for_robust_loss_function(J, f, rho)
    else:
        cost = 0.5 * np.dot(f, f)

    g = compute_grad(J, f)

    jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        scale, scale_inv = x_scale, 1 / x_scale

    Delta = norm(x0 * scale_inv)
    if Delta == 0:
        Delta = 1.0

    reg_term = 0
    damp = 0.0
    lsmr_maxiter = min(m, n)

    if max_nfev is None:
        max_nfev = x0.size * 100

    termination_status = None
    actual_reduction = None

    while True:
        g_norm = norm(g, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if termination_status is not None or nfev == max_nfev:
            break

        d = scale
        g_h = d * g

        J_h = _RightScaled(J, d)
        a, b = build_quadratic_1d(J_h, g_h, -g_h)
        to_tr = Delta / norm(g_h)
        ag_value = minimize_quadratic_1d(a, b, 0, to_tr)[1]
        reg_term = -ag_value / Delta**2

        damp_full = (damp**2 + reg_term)**0.5
        gn_h, _ = _lsmr_scaled(J.indptr, J.indices, J.data, m, n,
                               np.ascontiguousarray(d, dtype=float),
                               np.ascontiguousarray(f, dtype=float),
                               float(damp_full), 1e-6, 1e-6, 1e8, lsmr_maxiter)
        S = np.vstack((g_h, gn_h)).T
        S, _ = qr(S, mode='economic')
        JS = J_h.dot(S)
        B_S = np.dot(JS.T, JS)
        g_S = S.T.dot(g_h)

        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            p_S, _ = solve_trust_region_2d(B_S, g_S, Delta)
            step_h = S.dot(p_S)

            predicted_reduction = -evaluate_quadratic(J_h, g_h, step_h)
            step = d * step_h
            x_new = x + step
            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step_h)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue

            if loss_function is not None:
                cost_new = loss_function(f_new, cost_only=True)
            else:
                cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new

            Delta_new, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction,
                step_h_norm, step_h_norm > 0.95 * Delta)

            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
            if termination_status is not None:
                break

            Delta = Delta_new

        if actual_reduction > 0:
            x = x_new
            f = f_new
            cost = cost_new

            J = jac(x)

            if loss_function is not None:
                rho = loss_function(f)
                J, f = scale_for_robust_loss_function(J, f, rho)

            g = compute_grad(J, f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
        else:
            actual_reduction = 0

    if termination_status is None:
        termination_status = 0
    return OptimizeResult(x=x, status=termination_status, nfev=nfev)


class _CachedFun:
    """Mimics scipy's VectorFunction caching: a call at the same x as the
    previous one returns the cached value without calling the user function
    again, and the user function always receives a private copy of x.

    The Jacobian is handed out as a copy only when the solver will scale it in
    place (robust loss); with the linear loss it is only read."""

    def __init__(self, fun, jac, x0, kwargs, copy_jac=True):
        self._fun, self._jac, self._kw = fun, jac, kwargs
        self._copy_jac = copy_jac
        self.x = np.array(x0, dtype=float)
        self.f = np.atleast_1d(self._fun(self.x.copy(), **kwargs))
        self.J = self._as_csr(self._jac(self.x.copy(), **kwargs))
        self.f_ok = self.J_ok = True

    @staticmethod
    def _as_csr(J):
        return J if isinstance(J, sps.csr_array) else sps.csr_array(J)

    def _update_x(self, x):
        if not np.array_equal(x, self.x):
            self.x = np.array(x, dtype=float)
            self.f_ok = self.J_ok = False

    def fun(self, x):
        self._update_x(x)
        if not self.f_ok:
            self.f = np.atleast_1d(self._fun(self.x.copy(), **self._kw))
            self.f_ok = True
        return self.f.copy()

    def jac(self, x):
        self._update_x(x)
        if not self.J_ok:
            self.J = self._as_csr(self._jac(self.x.copy(), **self._kw))
            self.J_ok = True
        return self.J.astype(self.J.dtype) if self._copy_jac else self.J


def least_squares(fun, x0, jac, method='trf', ftol=1e-8, xtol=1e-8,
                  gtol=1e-8, x_scale=None, loss='linear', max_nfev=None,
                  kwargs=None, tr_solver='lsmr', **scipy_only):
    """``scipy.optimize.least_squares(method='trf', tr_solver='lsmr')`` for a
    callable sparse Jacobian and no bounds.  Returns an OptimizeResult
    (``x``, ``status``, ``nfev``; the full SciPy result on the fallback path).

    Extra keyword arguments (e.g. ``jac_sparsity``, which SciPy ignores for a
    callable Jacobian) are accepted and passed to SciPy on the fallback path.
    """
    kwargs = {} if kwargs is None else kwargs
    supported = (method == 'trf' and tr_solver in (None, 'lsmr') and callable(jac)
                 and loss in ('linear', 'soft_l1') and not scipy_only.keys() - {'jac_sparsity'})
    if not (ENABLED and supported):
        return _scipy_least_squares(
            fun, x0, jac=jac, method=method, ftol=ftol, xtol=xtol, gtol=gtol,
            x_scale=x_scale, loss=loss, max_nfev=max_nfev, kwargs=kwargs,
            tr_solver=tr_solver, **scipy_only)

    x0 = np.atleast_1d(x0).astype(float)
    ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol, 'trf')
    if isinstance(x_scale, str):
        if x_scale != 'jac':
            raise ValueError("x_scale must be 'jac' or array_like")
    else:
        x_scale = np.resize(np.asarray(1.0 if x_scale is None else x_scale,
                                       dtype=float), x0.shape)
    # make_strictly_feasible with infinite bounds is a plain copy.
    x0 = x0.copy()

    vf = _CachedFun(fun, jac, x0, kwargs, copy_jac=(loss != 'linear'))
    f0 = vf.fun(x0)
    J0 = vf.jac(x0)
    if not np.all(np.isfinite(f0)):
        raise ValueError("Residuals are not finite in the initial point.")
    loss_function = construct_loss_function(f0.size, loss, 1.0)
    return _trf_no_bounds_lsmr(vf.fun, vf.jac, x0, f0, J0, ftol, xtol, gtol,
                               max_nfev, x_scale, loss_function)
