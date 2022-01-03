import sys
import numpy as np
from analysis.params import ParamRun
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.data import DataManager
from model.theory import get_theory
import matplotlib.pyplot as plt
from model.power_spectrum import HalomodCorrection, hm_bias, hm_mean_mass
from model.utils import selection_planck_erf, selection_planck_tophat
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import rc
import pyccl as ccl
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# Theory predictor wrapper
class thr(object):
    def __init__(self, d):
        self.d = d

    def th(self, pars):
        return get_theory(p, self.d, cosmo, hm_correction=hm_correction,
                          selection=sel, **pars)

    def th1h(self, pars):
        return get_theory(p, self.d, cosmo, hm_correction=hm_correction,
                          selection=sel, include_2h=False, include_1h=True,
                          **pars)

    def th2h(self, pars):
        return get_theory(p, self.d, cosmo, hm_correction=hm_correction,
                          selection=sel, include_2h=True, include_1h=False,
                          **pars)

zmeans = []
bmeans = []
sbmeans = [[],[]]  # min and max error bar
zedges = np.array([0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05])
f = plt.figure(figsize=(8, 12))
gs_main = GridSpec(6, 2, figure=f)
imax = 6
extra = True
sr = 100000
for ibin in range(imax):
    fname_params = f'params_hybrid/params_spt/params_maglim_y21_bin{ibin}_minvar_cibcorr.yml'
    fname_params2 = f'params_hybrid/params_spt/params_maglim_y21_bin{ibin}_minvar.yaml' 
    p = ParamRun(fname_params)
    pp = ParamRun(fname_params2)
    run_name = p.get('mcmc')['run_name']

    # Cosmology (Planck 2018)
    cosmo = p.get_cosmo()

    # Include halo model correction if needed
    if p.get('mcmc').get('hm_correct'):
        hm_correction = HalomodCorrection(cosmo)
    else:
        hm_correction = None

    # Include selection function if needed
    sel = p.get('mcmc').get('selection_function')
    if sel is not None:
        if sel == 'erf':
            sel = selection_planck_erf
        elif sel == 'tophat':
            sel = selection_planck_tophat
        elif sel == 'none':
            sel = None

    for v, v2 in zip(p.get('data_vectors'), pp.get('data_vectors')):
        print(v['name'])

        # Construct data vector and covariance
        d = DataManager(p, v, cosmo, all_data=False)
        d_all = DataManager(p, v, cosmo, all_data=True)
        dp = DataManager(pp, v2, cosmo, all_data=True)
        thd = thr(d)
        thg = thr(d_all)
        thp = thr(dp)
        z, nz = np.loadtxt(d.tracers[0][0].dndz, unpack=True)
        zmean = np.average(z, weights=nz)
        sigz = np.sqrt(np.sum(nz * (z - zmean)**2) / np.sum(nz))
        zmeans.append(zmean)

        # Set up likelihood
        likd = Likelihood(p.get('params'), d.data_vector, d.covar, thd.th,
                         debug=p.get('mcmc')['debug'])

        likg = Likelihood(p.get('params'), d_all.data_vector, d_all.covar, thg.th,
                         debug=p.get('mcmc')['debug'])

        likp = Likelihood(pp.get('params'), dp.data_vector, dp.covar, thp.th,
                         debug=pp.get('mcmc')['debug'])
        # Set up sampler
        sam = Sampler(likd.lnprob, likd.p0, likd.p_free_names,
                      p.get_sampler_prefix(v['name']), p.get('mcmc'))

        # Read chains and best-fit
        sam.get_chain()
        sam.update_p0(sam.chain[np.argmax(sam.probs)])
        params = likd.build_kwargs(sam.p0)
        # Compute galaxy bias
        zarr = np.linspace(zmean - sigz, zmean + sigz, 10)
        mchain = np.array([hm_mean_mass(cosmo, 1./(1 + zarr), d.tracers[0][0].profile,
                          **(likd.build_kwargs(p0))) for p0 in sam.chain[::sr]])
        bgchain = np.array([hm_bias(cosmo, 1./(1 + zarr), d.tracers[0][0].profile,
                          **(likd.build_kwargs(p0))) for p0 in sam.chain[::sr]])
        bychain = np.array([hm_bias(cosmo, 1./(1 + zarr), d.tracers[1][1].profile,
                          **(likd.build_kwargs(p0))) for p0 in sam.chain[::sr]])
        bgmin, bg, bgmax = np.percentile(bgchain, [16, 50, 84])
        bymin, by, bymax = np.percentile(bychain, [16, 50, 84])
        mmin, mbest, mmax = np.percentile(mchain, [16, 50, 84])

        # Get effective number of degrees of freedom
        corr_mat = np.corrcoef(sam.chain, rowvar=False)
        _eigv, _eigvec = np.linalg.eig(corr_mat)
        _eigv[ _eigv > 1.0 ] = 1.
        _eigv[ _eigv < 0.0 ] = 0.
        #
        Ntot = len(_eigv)
        Neff = int(Ntot - np.sum( _eigv ))
        Neff = len(likd.dv) - Neff
        print('Effective ndof:', Neff)
        chi2 = likd.chi2(sam.p0)
        # SNR gg
        nbpw = len(d.ells[0])
        snr_gg = np.sqrt(np.einsum('i, ij, j', likd.dv[:nbpw], 
                         likd.ic[:nbpw, :nbpw],
                         likd.dv[:nbpw]))
        # SNR gy
        snr_gy = np.sqrt(np.einsum('i, ij, j', likd.dv[nbpw:],
                         likd.ic[nbpw:, nbpw:],
                         likd.dv[nbpw:]))
        print('SNR gg', snr_gg)
        print('SNR gy', snr_gy) 
        lmin = v["twopoints"][0]["lmin"]
        chi = ccl.comoving_radial_distance(cosmo, 1/(1+zmean))
        kmax = p.get("mcmc")["kmax"]
        lmax = kmax*chi - 0.5
        th_all = thg.th(params).reshape((2, -1))
        th_1h = thg.th1h(params).reshape((2, -1))
        th_2h = thg.th2h(params).reshape((2, -1))
        ev = np.sqrt(np.diagonal(likg.cv)).reshape((2, -1))
        mydv = likg.dv.reshape((2, -1))
        evp = np.sqrt(np.diagonal(likp.cv)).reshape((2, -1))
        mydvp = likp.dv.reshape((2, -1))
        th_p = thp.th(params).reshape((2, -1))
        for igrid in range(2):
            gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[ibin, igrid],
                                         height_ratios=[3, 1], hspace=0)
            ax1 = f.add_subplot(gs[0])
            ax2 = f.add_subplot(gs[1])
            res = (mydv[igrid, :] - th_all[igrid, :])/ev[igrid, :]
            res2 = (mydvp[igrid, :] / th_p[igrid, :] - 1) # /evp[igrid, :]
            print('size of dvs', len(d_all.ells[0]), res.shape, len(mydv[igrid, :]))
            ax2.axhline(color="k", ls="--")
            ax2.errorbar(d_all.ells[0], res, yerr=np.ones_like(mydv[igrid, :]), fmt="r.") 
            #ax2.errorbar(d_all.ells[0], res2, yerr=evp[igrid, :]/th_p[igrid, :], fmt="k.", alpha=0.2)
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax2.set_xscale("log")
            if ibin >= 4:
                ax1.axvspan(0, d_all.ells[0][-1]*1.2, alpha=0.2, color='k')
                ax2.axvspan(0, d_all.ells[0][-1]*1.2, alpha=0.2, color='k')
            ax1.set_xlim(d_all.ells[0][0]/1.1, d_all.ells[0][-1]*1.1)
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim(-2.7, 2.7)        
            #ax2.set_ylim(-1, 1)
            # plot data & theory
            ax1.plot(d_all.ells[0], th_1h[igrid, :], ls="--", c="k", alpha=0.3,
                     label=r"$\mathrm{1}$-$\mathrm{halo}$")

            ax1.plot(d_all.ells[0], th_2h[igrid, :], ls="-.", c="k", alpha=0.3,
                 label=r"$\mathrm{2}$-$\mathrm{halo}$")

            ax1.plot(d_all.ells[0], th_all[igrid, :], ls="-", c="k", label=r"$\mathrm{1h+2h}$")
            # Adding Planck's model
            #ax1.plot(d_all.ells[0], th_p[igrid, :], ls="-", c="k", label=r"$\mathrm{1h+2h}$")
            ax1.errorbar(d_all.ells[0], mydv[igrid, :], yerr=ev[igrid, :], fmt="r.")
            if extra:
                ax1.errorbar(dp.ells[0], mydvp[igrid, :], yerr=evp[igrid, :], fmt="k.", alpha=0.2)
            ax1.axvspan(0, lmin, alpha=0.3, color='k')
            ax2.axvspan(0, lmin, alpha=0.3, color='k')
            ax1.axvspan(lmax, 1.1*d_all.ells[0][-1], alpha=0.3, color='k')
            ax2.axvspan(lmax, 1.1*d_all.ells[0][-1], alpha=0.3, color='k')

            if igrid == 0:
                ax1.text(0.6, 0.8, "$\\chi^2/\\rm{ndof}=%.2lf/%d$" %
                     (chi2, Neff), transform=ax1.transAxes)
                ax1.text(0.6, 0.6, "$%.2f < z < %.2f$" %(zedges[ibin], zedges[ibin+1]),
                         transform=ax1.transAxes)
                ax1.text(0.2, 0.2, "$SNR^{gg} = %.1f$" % snr_gg, transform=ax1.transAxes)
            if igrid == 1:
                ax1.text(0.7, 0.7, "$SNR^{gy} = %.1f$" % snr_gy, transform=ax1.transAxes)

            ax1.set_ylabel('$C_\\ell$', fontsize=15)
            ax2.set_ylabel('$\\Delta_\\ell$', fontsize=15)
 
            if igrid == 0:
               ax1.set_ylim(3e-8, 2e-4) 
            if igrid == 1:
               ax1.set_ylim(5e-16, 3e-10)
            if ibin == 0:
                if igrid == 0:
                    ax1.text(0.45, 1.1, r"$C_{\ell}^{gg}$", fontsize=18,
                             transform=ax1.transAxes)
                if igrid == 1:
                    ax1.text(0.45, 1.1, r"$C_{\ell}^{gy}$", fontsize=18,
                             transform=ax1.transAxes)
                    ax1.legend(loc="lower center", ncol=4, fontsize=8,
                               borderaxespad=0.1, columnspacing=1.9)

            if ibin != 5:
                ax2.get_xaxis().set_visible(False)
            else:
                ax2.set_xlabel('$\\ell$', fontsize=15)
        print(" Best-fit parameters:")
        pars = []
        for i, nn, in enumerate(sam.parnames):
            CHAIN = sam.chain[:, i]
            vmin, vv, vmax = np.percentile(CHAIN, [16, 50, 84])
            pars.append(vv)
            errmin, errmax = vv-vmin, vmax-vv
            print("  " + nn + " : %.3lE +/- (%.3lE %.3lE)" % (vv, errmax, errmin))
            if nn == 'b_hydro':
                bmeans.append(vv)          # median
                sbmeans[0].append(errmin)  # min errorbar
                sbmeans[1].append(errmax) # max errorbar
            chain = sam.chain
        pars.append(likd.chi2(sam.p0))
        pars.append(len(d.data_vector))
        np.save(p.get_outdir() + "/best_fit_params_" + run_name + "_"
                +v["name"]+".npy", np.array(pars))
        print(" chi^2 = %lf" % (likd.chi2(sam.p0)))
        print(" ndof = %d" % (Neff))
        print(" b_g = %.3lE +/- (%.3lE %.3lE) " % (bg, bg-bgmin, bgmax-bg))
        print(" b_y = %.3lE +/- (%.3lE %.3lE) " % (by, by-bymin, bymax-by))
        print(" M_h = %.3lE +/- (%.3lE %.3lE) " % (mbest, mbest-mmin, mmax-mbest))
        #print(" b_H = %.3lE +/- (%.3lE %.3lE) " % (vv, vv-errmin, vv-errmax))

f.tight_layout(h_pad=0.05, w_pad=0.1)
f.show()
f.savefig("fits_minvar_cibcorr.pdf", bbox_inches="tight")
