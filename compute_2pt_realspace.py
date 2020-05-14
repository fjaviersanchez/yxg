import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import fitsio
from argparse import ArgumentParser
from scipy.stats import binned_statistic
import treecorr as tc
import os
import astropy.table
from kmeans_radec import KMeans, kmeans_sample

parser = ArgumentParser()
parser.add_argument('--input-galaxies', '-i', dest='input_galaxies', type=str, nargs='+', default=None,
    help='Path of the galaxy catalog file(s)')
parser.add_argument('--input-ymap-path', dest='y_path', type=str, default=None,
    help='Path to the ymap file')
parser.add_argument('--output-file', dest='out_path', type=str, default=None,
    help='Path of the output file')
parser.add_argument('--debug', dest='debug', action='store_true', help='Show debugging plots', default=False)
parser.add_argument('--mask-path', dest='mask_path', default=None, type=str, help='Path to galaxy mask')
parser.add_argument('--ymask-path', dest='ymask_path', default=None, type=str, help='Path to y-map mask')
parser.add_argument('--weight-path', dest='weight_path', default=None, type=str, help='Path to weight map', nargs='+')
parser.add_argument('--use-maps', dest='make_maps', default=False, action='store_true',
     help='--use-maps uses maps for the 2pt correlations instead of the positions of the sources')
parser.add_argument('--n-jk', dest='n_jk', default=100, type=int, help='Number of jackknife regions')
args = parser.parse_args()

def maskdata(data, mask):
    pxnums = hp.ang2pix(hp.get_nside(mask), data['RA'], data['DEC'], lonlat=True)
    goodpx = np.in1d(pxnums, np.where(mask)[0])
    return goodpx

def make_hp_map(nside, input_data, weights=None):
    _px_nums = hp.ang2pix(nside, input_data['RA'], input_data['DEC'], lonlat=True)
    _counts = np.bincount(_px_nums, weights=weights, minlength=hp.nside2npix(nside)).astype(float)
    return _counts

def make_rnd_map(mask, nrand, weight=True):
    px_rnd = np.random.choice(np.where(mask>0)[0], size=nrand)
    map_rnd = np.bincount(px_rnd, minlength=len(mask)).astype(float)
    if weight:
        map_rnd[mask>0]=map_rnd[mask>0]/mask[mask>0]
    return map_rnd

def make_rnd(data, mask, nrnd):
    ra = data['RA']
    dec = data['DEC']
    min_ra = np.min(ra)
    max_ra = np.max(ra)
    min_cth = np.min(np.sin(np.radians(dec)))
    max_cth = np.max(np.sin(np.radians(dec)))
    rnd_ra = np.random.uniform(min_ra, max_ra, size=nrnd)
    rnd_dec = np.degrees(np.arcsin(np.random.uniform(min_cth, max_cth, size=nrnd)))
    _pxnums = hp.ang2pix(hp.get_nside(mask), rnd_ra, rnd_dec, lonlat=True)
    _mask = np.in1d(_pxnums, np.where(mask>0)[0])
    cat_rnd = tc.Catalog(ra = rnd_ra[_mask], dec = rnd_dec[_mask], ra_units='deg', dec_units='deg')
    return cat_rnd

def setup_tc_catalogs(data, ymap, mask_galaxy, weight_map, make_map=False):
    nside = hp.get_nside(mask_galaxy)
    if make_map:
        ra ,dec = hp.pix2ang(nside, np.where(mask_galaxy)[0], lonlat=True)
        map_galaxy = make_hp_map(hp.get_nside(mask_galaxy),  data)*weight_map 
        cat_galaxy = tc.Catalog(ra=ra, dec=dec, w=map_galaxy[mask_galaxy>0], ra_units='deg', dec_units='deg')
        map_rnd = make_rnd_map(mask_galaxy, int(10*np.sum(map_galaxy)))
        cat_rnd = tc.Catalog(ra=ra, dec=dec, w=map_rnd[mask_galaxy>0], ra_units='deg', dec_units='deg')
    else:
        ra = data['RA']
        dec = data['DEC']
        pxnum = hp.ang2pix(nside, ra, dec, lonlat=True) 
        print(f'Preparing {len(ra)} objects catalog.')
        cat_galaxy = tc.Catalog(ra = ra, dec = dec, w = weight_map[pxnum], ra_units='deg', dec_units='deg')
        cat_rnd = make_rnd(data, mask_galaxy, 10*len(data['RA']))
    ra ,dec = hp.pix2ang(nside, np.where(mask_galaxy)[0], lonlat=True)
    cat_y = tc.Catalog(ra=ra, dec=dec, k=ymap[mask_galaxy>0], ra_units='deg', dec_units='deg')
    return cat_galaxy, cat_y, cat_rnd

def compute_corr(cat_galaxy, cat_y, cat_rnd, min_sep=1., max_sep=100, nbins=10):
    dy = tc.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
    ry = tc.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
    ry.process(cat_rnd, cat_y)
    dy.process(cat_galaxy, cat_y)
    xi, _ = dy.calculateXi(ry)
    return np.exp(dy.meanlogr), xi

def compute_auto(data, mask, nrnd, min_sep=1., max_sep=100., nbins=10):
    _pxnums = hp.ang2pix(hp.get_nside(mask), data['RA'], data['DEC'], lonlat=True)
    _mask = np.in1d(_pxnums, np.where(mask>0)[0])
    cat_galaxy = tc.Catalog(ra = data['RA'][_mask], dec = data['DEC'][_mask], ra_units='deg', dec_units='deg')
    cat_rnd = make_rnd(data, mask, nrnd)
    dd = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
    dr = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
    rr = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
    dd.process(cat_galaxy)
    dr.process(cat_galaxy, cat_rnd)
    rr.process(cat_rnd)
    xi, varxi = dd.calculateXi(rr, dr)
    return np.exp(dd.meanlogr), xi

def get_kmeans_labels(data, njk):
    X = np.zeros((len(data['RA']), 2))
    X[:,0] = data['RA']
    X[:,1] = data['DEC']
    km = kmeans_sample(X, njk, maxiter=100, tol=1e-4)
    print('KMeans info:', km.converged, np.bincount(km.labels))
    return km

ymap = hp.read_map(args.y_path)
mask_y = hp.read_map(args.ymask_path)
data_out = dict()
mask_galaxies = np.zeros(len(mask_y), dtype=float)
data_mask = fitsio.read(args.mask_path)
mask_galaxies[data_mask['HPIX']] = data_mask['FRACGOOD']
mask_galaxies = mask_galaxies*mask_y # Compute the correlations in the overlapping region

print(f'Going to compute {len(args.input_galaxies)} z-bins')
data_cov = dict()
for i in range(len(args.input_galaxies)):
    print('Bin', i)
    data = fitsio.read(args.input_galaxies[i])
    data = data[maskdata(data, mask_galaxies)] # Mask the data

    if args.weight_path is not None:
        wgt_map = np.zeros(len(mask_galaxies), dtype=float)
        wgt_data = fitsio.read(args.weight_path[i])
        wgt_map[wgt_data['HPIX']] = wgt_data['VALUE']
    else:
        wgt_map = np.ones(len(mask_galaxies), dtype=float)

    if args.debug:
        hp.mollview(wgt_map)
        hp.mollview(ymap)
        hp.mollview(make_hp_map(hp.get_nside(ymap), data))
        plt.show()
    # Compute correlation function
    cat_galaxy, cat_y, cat_rnd = setup_tc_catalogs(data, ymap, mask_galaxies, wgt_map, args.make_maps)
    theta, w = compute_corr(cat_galaxy, cat_y, cat_rnd)
    _, w_auto = compute_auto(data, mask_galaxies, 10*len(data['RA']))
    if args.debug:
        plt.figure()
        plt.loglog(theta, w, 'o')
        plt.loglog(theta, w_auto, 'o')
        plt.show()
    data_out['theta'] = theta
    data_out[f'w_gy_{i}'] = w
    data_out[f'w_gg_{i}'] = w_auto
    # Compute Jackknife
    if args.n_jk > 0:
        _km = get_kmeans_labels(data, args.n_jk)
        labels = _km.labels
        X2 = np.zeros((np.count_nonzero(mask_galaxies), 2))
        _ra, _dec = hp.pix2ang(hp.get_nside(mask_galaxies), np.where(mask_galaxies>0)[0], lonlat=True)
        X2[:,0] = _ra
        X2[:,1] = _dec 
        labels2 = _km.find_nearest(X2)
        if args.debug:
            print('Labels', np.unique(labels))
            print('Labels 2', np.bincount(labels2), np.unique(labels2))
        for i_jk in range(0,args.n_jk):
            print(f'JK: {i_jk}, {np.count_nonzero(labels!=i_jk)} galaxies left')
            mask_galaxies_aux = np.zeros_like(mask_galaxies)
            jk_reg_px = np.unique(hp.ang2pix(hp.get_nside(mask_galaxies), data['RA'][labels==i_jk], data['DEC'][labels==i_jk], lonlat=True))
            mask_galaxies_aux[mask_galaxies>0] = mask_galaxies[mask_galaxies>0]
            mask_galaxies_aux[jk_reg_px] = 0.
            if args.debug:
                print('Pixels matching', len(jk_reg_px))
                hp.mollview(mask_galaxies_aux)
                hp.mollview(mask_galaxies_aux-mask_galaxies)
                plt.show()
            cat_galaxy, cat_y, cat_rnd = setup_tc_catalogs(data[labels!=i_jk], ymap, mask_galaxies_aux, wgt_map, args.make_maps)
            theta, w = compute_corr(cat_galaxy, cat_y, cat_rnd)
            _, w_auto = compute_auto(data[labels!=i_jk], mask_galaxies_aux, 10*len(data['RA'][labels!=i_jk]))
            data_cov[f'w_gy_{i}_{i_jk}'] = w
            data_cov[f'w_gg_{i}_{i_jk}'] = w_auto
tab = astropy.table.Table(data_out)
tab.write(args.out_path, overwrite=True)
data_cov['theta'] = theta
tab2 = astropy.table.Table(data_cov)
tab2.write(args.out_path.replace('.fits', '_cov.fits'), overwrite=True) 
