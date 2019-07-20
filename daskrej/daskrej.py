#!/usr/bin/env python

import logging
import os
import time

from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
import dask
import dask.array as da

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import sep


logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s')


LOG = logging.getLogger()

LOG.setLevel(logging.INFO)


_OUTPUT_DIR = os.path.join(
    '/',
    *os.path.dirname(os.path.abspath(__file__)).split('/')[:-1],
    'daskrej_output'
)



@dask.delayed
def read_file(fname, extname='sci', extnums=(1, 2)):
    """ Grab the data from extensions named EXT from FITS file

    Parameters
    ----------
    extname : str
        Name of extension to extract data from (e.g. 'SCI' or 'DQ')

    extnums : tuple
        Tuple of the extension numbers. This should always be a tuple, even
        if it just contains one element
    """
    ext_tuples = [(extname, num) for num in extnums]
    ext_data = []
    with fits.open(fname) as hdu:
        for val in ext_tuples:
            try:
                ext = hdu.index_of(val)
            except KeyError:
                LOG.warning('{} is missing for {}'.format(val, fname))
            else:
                ext_data.append(hdu[ext].data)
    return np.concatenate(ext_data, axis=0)


@dask.delayed
def compute_sky_sep(data, fname, bw=None, bh=None, fw=None, fh=None):
    """

    Parameters
    ----------
    data :

    bw :
        Width of the box to compute the background in

    bh :
        Height of the box to compute the background in

    fw :
        Width of the filter applied

    fh :
        Height of the filter applied

    Returns
    -------

    """
    try:
        bkg = sep.Background(data, bw=bw, bh=bh, fw=fw, fh=fh)
    except ValueError as e:
        msg = (
            '{}\n'
            'Reformatting image with .byteswap().newbyteorder()'.format(e)
        )
        LOG.info(msg)
        bkg = sep.Background(data.byteswap().newbyteorder(),
                             bw=bw, bh=bh, fw=fw, fh=fh)
    # msg = (
    #     'fname: {}\n'
    #     'global background level: {:.3f}\n'
    #     'global background RMS: {:.3f}\n'
    #     '{}\n'.format(fname,
    #                   bkg.globalback,
    #                   bkg.globalrms,
    #                   '-' * 79)
    # )
    # LOG.info(msg)
    return bkg

@dask.delayed
def noise_model(data, rdnoise, sigma_thresh):
    estimated_variance = sigma_thresh**2 * (rdnoise**2 + data)
    return estimated_variance

@dask.delayed
def img_noise(data, bkg, init_guess):
    """

    Parameters
    ----------
    data :
        `dask.array` for SCI extensions of the input images [ELECTRONS]

    bkg :
        `numpy.array` representing the background for the input image

    init_guess :
        `dask.array` representing the initial guess at the final
        cosmic-ray-cleaned image

    Returns
    -------

    """
    variance = da.square(data - (bkg.globalback + init_guess))
    return variance


@dask.delayed
def generate_cr_mask(variance, expected_variance, fill_val=8192):
    cr_mask = da.where(variance > expected_variance, fill_val, 0)
    return cr_mask


@dask.delayed
def combine_stack(lazy_array, bkg, cr_mask):
    final_array = da.where(cr_mask, bkg.globalback, lazy_array)
    return final_array

def mk_fig(show_axis_labels=True, show_grid=False):
    """Generate a figure with two axes for plotting the label

    Parameters
    ----------
    show_axis_labels : bool
        Show major tick labels on both axes

    show_grid : bool
        Show a grid

    Returns
    -------
    fig : matplotlib.figure.Figure
        Instance of a matplotlib Figure object

    ax1 : matplotlib.axes.Axes
        Instance of a maplotlib Axes object

    ax2 : matplotlib.axes.Axes
        Instance of a matplotlib Axes object


    """
    grid = plt.GridSpec(1, 2, wspace=0.2, hspace=0.15)
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1], sharex=ax1, sharey=ax1)
    for ax in [ax1, ax2]:
        if show_grid:
            ax.grid(False)

        if not show_axis_labels:
            ax.tick_params(axis='both', bottom=False,
                           labelbottom=False,
                           left=False,
                           labelleft=False)
    return fig, ax1, ax2


def plot(data, cr_mask, xlim=None, ylim=None, fout=None, save=False):
    """ Plot the label

    Parameters
    ----------
    xlim : tuple
        Limits for the x-axis

    ylim : tuple
        Limits for the y-axis

    fout : str
        Filename to save the image to (e.g. example_mask.png)

    save : bool
        If True, save the generated plot to the plots directory


    Returns
    -------


    """

    fig, ax1, ax2 = mk_fig()


    norm = ImageNormalize(data,
                          stretch=LinearStretch(),
                          interval=ZScaleInterval())

    ax1.imshow(data, norm=norm, cmap='gray', origin='lower')
    ax2.imshow(cr_mask, cmap='gray', interpolation='nearest', origin='lower')

    if xlim is not None:
        ax1.set_xlim(xlim)
    if ylim is not None:
        ax1.set_ylim(ylim)

    ax1.set_title('SCI Extension')
    ax2.set_title('Cosmic Ray Segmentation Map')
    # if save:
    #     fig.savefig(fout,
    #                 format='png',
    #                 dpi=300, bbox_inches='tight')

    plt.show()

@dask.delayed
def write_image(exthdr=None, extdata=None, crmask=None, fout=None):
    hdu_list = fits.HDUList()
    hdu_list.append(fits.ImageHDU(extdata, header=exthdr))
    hdu_list.append(fits.ImageHDU(crmask, header=exthdr))
    hdu_list.writeto(fout, overwrite=True)
    return fout

def compute_initial_guess(lazy_arrays,
                          num_chunks=None,
                          combine_func=np.median):

    lazy_stack = da.stack(lazy_arrays, axis=0)
    chunk_dimensions = (
        lazy_stack.shape[0],
        int(lazy_stack.shape[1] / num_chunks),
        int(lazy_stack.shape[2] / num_chunks)
    )

    # Rechunk the stack to make the computation more memory efficient
    rechunked_stack = lazy_stack.rechunk(chunks=chunk_dimensions)

    # Compute the initial guess
    init_guess = da.map_blocks(combine_func,
                               rechunked_stack,
                               axis=0,
                               chunks=(rechunked_stack.chunksize[1],
                                       rechunked_stack.chunksize[2]),
                               drop_axis=0).compute()

    return init_guess


def iterative_clean(
        lazy_arrays,
        shape=None,
        dtype=None,
        num_chunks=None,
        flist=None,
        sigma_thresh=None,
        bkg_list=None,
        iteration=None,
        init_guess=None,
        cr_masks=None
):
    """ Perform a single iteration of the CR cleaning process

    Parameters
    ----------
    lazy_arrays

    Returns
    -------

    """
    if init_guess is None:
        # Compute the initial guess
        init_guess = compute_initial_guess(lazy_arrays,
                                           num_chunks=num_chunks,
                                           combine_func=np.median)

    # Compute the expected variance for a given pixel using the initial
    # guess of the final cosmic-ray-cleaned image
    expected_variance = noise_model(data=init_guess,
                                    rdnoise=4.3,
                                    sigma_thresh=sigma_thresh)

    if bkg_list is None:
        # Compute the background of each image
        bkg_list = [
            compute_sky_sep(data=data, fname=flist[j],  bw=64, bh=64, fw=3, fh=3)
            for j, data in enumerate(lazy_arrays)
        ]

    # new_cr_masks = []
    if cr_masks is None:
        cr_masks=[]
        propagate = False
    else:
        propagate = True

    # For each image, compute the observed variance and compare with expected
    for i, (data, bkg) in enumerate(zip(lazy_arrays, bkg_list)):
        variance = img_noise(data=data, bkg=bkg, init_guess=init_guess)
        mask = generate_cr_mask(variance=variance,
                                expected_variance=expected_variance)
        if propagate:
            cr_masks[i] += mask
        else:
            cr_masks.append(mask)



    cleaned_arrays = []
    for lazy, bkg, mask in zip(lazy_arrays, bkg_list, cr_masks):
        cleaned = combine_stack(lazy_array=lazy,
                                bkg=bkg,
                                cr_mask=mask)
        cleaned_arrays.append(cleaned)

    # cleaned_arrays = [
    #     da.from_delayed(x, shape=shape, dtype=dtype) for x in cleaned_arrays
    # ]
    cleaned_arrays = list(dask.compute(*cleaned_arrays))
    return cleaned_arrays, cr_masks

# @profile
def daskrej(
        flist,
        shape=None,
        dtype=None,
        num_chunks=8,
        sigma_thresh=(10,)):
    """

    Parameters
    ----------
    flist
    shape
    dtype
    num_chunks

    Returns
    -------

    """
    st = time.time()
    msg = (
        'Starting process for {} images\n '
        'sigma thresholds = [{}] \n'
        'number of chunks = {}'.format(
            len(flist),
            ','.join([str(val) for val in sigma_thresh]),
            num_chunks
        )
    )
    LOG.info(msg)
    # Read in a sample dataset to determine the shape and dtype of the list
    #of images
    if shape is None and dtype is None:
        sample = read_file(flist[0]).compute()
        shape = sample.shape
        dtype = sample.dtype

    # Construct the graph for reading in the FITS data
    delayed_reads = [
        read_file(f, extname='SCI', extnums=(1,2)) for f in flist
    ]

    # Construct a list of arrays from the delayed reads
    lazy_arrays = [
        da.from_delayed(x, shape=shape, dtype=dtype) for x in delayed_reads
    ]

    LOG.info('Starting background determination')
    # Compute the background of each image
    bkg_list = [
        compute_sky_sep(data=data, fname=flist[j], bw=64, bh=64, fw=3, fh=3)
        for j, data in enumerate(lazy_arrays)
    ]

    # Turn the delayed background computations into lazy arrays
    # bkg_list = [
    #     da.from_delayed(bkg, shape=shape, dtype=dtype) for bkg in bkg_list
    # ]

    LOG.info('Finished background determination')
    init_guess = compute_initial_guess(lazy_arrays=lazy_arrays,
                                       num_chunks=num_chunks,
                                       combine_func=np.median)
    i=0
    while i < len(sigma_thresh):
        LOG.info('Running iteration {}, sigma={} ...'.format(i+1, sigma_thresh[i]))
        if i == 0:
            st_iter = time.time()
            cleaned_arrays, cr_masks = iterative_clean(
                lazy_arrays=lazy_arrays,
                shape=shape,
                dtype=dtype,
                num_chunks=num_chunks,
                flist=flist,
                bkg_list=bkg_list,
                init_guess=init_guess,
                sigma_thresh=sigma_thresh[i]
            )
            et_iter = time.time()
            duration = et_iter - st_iter
            units = 'seconds'
            if duration > 60:
                duration /= 60
                units = 'minutes'
            msg = (
                'Finished iteration {}. \n'
                'Total run time: {:.3f} {}'.format(i + 1, duration, units)
            )
            LOG.info(msg)
        else:
            st_iter = time.time()
            cleaned_arrays, cr_masks = iterative_clean(
                lazy_arrays=cleaned_arrays,
                shape=shape,
                dtype=dtype,
                num_chunks=num_chunks,
                flist=flist,
                sigma_thresh=sigma_thresh[i],
                bkg_list=bkg_list,
                init_guess=init_guess,
                cr_masks=cr_masks
            )
            et_iter = time.time()
            duration = et_iter - st_iter
            units = 'seconds'
            if duration > 60:
                duration /= 60
                units = 'minutes'
            msg = (
                'Finished iteration {}. \n'
                'Total run time: {:.3f} {}'.format(i + 1, duration, units)
            )
            LOG.info(msg)
        i += 1

    # imout = []
    # for cleaned, fname, crmask in zip(cleaned_arrays, flist, cr_masks):
    #     fout = fname.replace('_flt.fits','_daskrej_{}iter_flt.fits'.format(i))
    #     exthdr = fits.getheader(fname, ('sci', 1))
    #     imout.append(write_image(fout=fout,
    #                               exthdr=exthdr,
    #                               crmask=crmask,
    #                               extdata=cleaned)
    #                   )
    # flist_out = dask.compute(*imout, scheduler='threads')
    cleaned_stack = da.stack(cleaned_arrays, axis=0)

    # Compute the sum of all the good background pixels
    final_image = cleaned_stack.sum(axis=0).compute()

    # Write out the final image
    write_image(fout='{}/daskrej_crj.fits'.format(_OUTPUT_DIR),
                extdata=final_image,
                crmask=np.zeros(final_image.shape)).compute()

    et = time.time()
    # Comptue the duration and set the units accordingly
    duration = et - st
    units='seconds'
    if duration > 60:
        duration /= 60
        units='minutes'
    msg = (
        'Processed {} files. \n'
        'Total run time: {:.3f} {}'.format(len(flist), duration, units)
    )
    LOG.info(msg)


    return final_image

if __name__ == '__main__':
    import glob

    flist = glob.glob("/Users/nmiles/dask_acsrej/test_data/?????????_flt.fits")
    daskrej(flist,
            shape=(4096, 4096),
            dtype=np.float64,
            num_chunks=8,
            sigma_thresh=[8,6]
            )