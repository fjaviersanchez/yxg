import numpy as np
from .power_spectrum import hm_ang_power_spectrum
from .utils import beam_hpix, beam_gaussian
import pyccl as ccl
from time import time

def get_theory(p, dm, cosmo, theta, return_separated=False,
               include_1h=True, include_2h=True,
               selection=None,
               hm_correction=None, d_ell=24, lmax=11000, **kwargs):
    """Computes the theory prediction used in the MCMC.

    Args:
        p (:obj:`ParamRun`): parameters for this run.
        dm (:obj:`DataManager`): data manager for this set of
            correlations.
        cosmo (:obj:`ccl.Cosmology`): cosmology object.
        return_separated (bool): return output power spectra
            in separate arrays.
        hm_correction(:obj:`HalomodCorrection`): halo model correction
            factor.
        selection (function): selection function in (M,z) to include
            in the calculation. Pass None if you don't want to select
            a subset of the M-z plane.
        **kwargs: model parameters
    """
    nz_default = p.get('mcmc')['nz_points_g']
    use_zlog = p.get('mcmc')['z_log_sampling']

    ws_out = []
    #ells = np.logspace(0, np.log(lmax), np.log(lmax/d_ell)).astype(int)
    #ells = np.arange(0,lmax,d_ell)
    ells = np.unique(np.geomspace(1e-1, 30000, 75).astype(np.int))
    b_hpix = beam_hpix(ells, 4096)
    b0 = b_hpix*beam_gaussian(ells, 1.25) 
    for tr, ls, bms in zip(dm.tracers, dm.ells, dm.beams):
        profiles = (tr[0].profile, tr[1].profile)
        if tr[0].name == tr[1].name:
            zrange = tr[0].z_range
            zpoints = nz_default
        else:
            # At least one of them is g
            if tr[0].type == 'g' or tr[1].type == 'g':
                if tr[0].type != tr[1].type:  # Only one is g
                    # Pick which one is g.
                    # That one governs the redshift slicing
                    t = tr[0] if tr[0].type == 'g' else tr[1]
                    zrange = t.z_range
                    zpoints = nz_default
                else:  # Both are g, but different samples
                    # Get a range that encompasses both N(z) curves
                    zrange = [min(tr[0].z_range[0], tr[1].z_range[0]),
                              max(tr[0].z_range[1], tr[1].z_range[1])]
                    # Get the minimum sampling rate of both curves
                    dz = min((tr[0].z_range[1]-tr[0].z_range[0])/nz_default,
                             (tr[1].z_range[1]-tr[1].z_range[0])/nz_default)
                    # Calculate the point preserving that sampling rate
                    zpoints = int((zrange[1]-zrange[0])/dz)
            else:  # Only other option right now is for both of them to be y
                zrange = tr[0].z_range
                zpoints = nz_default
        #t0 = time()
        cl = hm_ang_power_spectrum(cosmo, ells, profiles,
                                   zrange=zrange, zpoints=zpoints,
                                   zlog=use_zlog, hm_correction=hm_correction,
                                   include_1h=include_1h,
                                   include_2h=include_2h,
                                   selection=selection,
                                   **kwargs)
        #t1 = time()
        #print('Time ellapsed (cls):', t1-t0)
        if cl is None:
            return None
        if tr[0].type == 'g' and tr[1].type == 'g':
            cl *= 1 # b_hpix**2 -> if the galaxies are in a map # The galaxies aren't in a map for real space so no beam is needed
        if tr[0].type == 'g' and tr[1].type == 'y':
            cl *= b0*b_hpix # The ymap has the beam + a 4096 healpix map smoothing and the galaxies don't have a beam
        else:
            cl *= b0**2
        #cl *= b0  # Multiply by beams
        #t0 = time()
        corr = ccl.correlation(cosmo, ells, cl, theta/60.) # Theta assumes scales in degrees (and we feed arcmins)
        #t1 = time()
        #print('Time ellapsed (corr):', t1-t0)
        #print(corr)
        if return_separated:
            ws_out.append(corr)
        else:
            ws_out += corr.tolist()
    return np.array(ws_out)
