import maid_almaz_version
import numpy as np


def maid_structure_func(particle_user, W, Q2, cos_theta):
    maid_particle = 'pi+ n' if particle_user == 'Pin' else 'pi0 p'
    interpolator = maid_almaz_version.InterpSigmaCorrectedCached('maid', maid_particle)
    R_T, R_L, R_TT, R_TL, R_TLp = np.ravel(interpolator.interp_R(W, Q2, cos_theta))
    ds_T, ds_L, ds_TT, ds_TL, ds_TLp = np.ravel(interpolator.interp_dsigma_comps(W, Q2, cos_theta))
    return R_T, R_L, R_TT, R_TL, R_TLp, ds_T, ds_L, ds_TT, ds_TL, ds_TLp


def maid_cross_section(particle_user, W, Q2, cos_theta, phi, Eb, h):
    maid_particle = 'pi+ n' if particle_user == 'Pin' else 'pi0 p'
    interpolator = maid_almaz_version.InterpSigmaCorrectedCached('maid', maid_particle)
    DCS = interpolator.interp_dsigma(W, Q2, cos_theta, phi, Eb, h)
    return DCS
