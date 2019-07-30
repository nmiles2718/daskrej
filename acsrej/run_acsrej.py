#!/usr/bin/env python

import argparse
import glob
import logging
import os
import time

from acstools import acsrej

parser = argparse.ArgumentParser()
parser.add_argument('-dir', type=str,
                    help='/path/to/directory/to/process/')


LOG = logging.getLogger()

LOG.setLevel(logging.INFO)

_OUTPUT_DIR = os.path.join(
    '/',
    *os.path.dirname(os.path.abspath(__file__)).split('/')[:-1],
    'acsrej_output'
)

_TEST_DATA = os.path.join(
'/',
    *os.path.dirname(os.path.abspath(__file__)).split('/')[:-1],
    'test_data'
)


def run_acsrej(exp_list, fout, crsigmas, initgues, skysub):
    acsrej.acsrej(exp_list,
                  output=fout,
                  crrejtab='/Users/nmiles/hst_cosmic_rays/data/ACS/29p1548cj_crr_WFC.fits',
                  crsigmas=crsigmas,
                  # exec_path=exec_path,
                  skysub=skysub,
                  initgues=initgues,
                  scalense=0.,
                  verbose=True)


def copy_files(flist, output_dir):
    LOG.info('Copying data products to output directory')
    for f in flist:
        os.system('cp -v {} {}/{}'.format(f,
                                          output_dir,
                                          os.path.basename(f)))


def prep_test_data(test_dir=None, output_dir=None):
    LOG.info('Copying over test data from {}'.format(test_dir))
    flist = glob.glob(test_dir+'/*flt.fits')
    copy_files(flist, output_dir)
    return output_dir


def combine_data(directory):
    st = time.time()
    dirname = prep_test_data(test_dir=_TEST_DATA,
                            output_dir=_OUTPUT_DIR)

    flt_flist = glob.glob(dirname +'/*flt.fits')

    run_acsrej(flt_flist,
               fout_long,
               crsigmas='8,6',
               initgues='med',
               skysub='mode')
    et = time.time()

    # Comptue the duration and set the units accordingly
    duration = et - st
    units = 'seconds'
    if duration > 60:
        duration /= 60
        units = 'minutes'
    msg = (
        'Processed {} files. \n'
        'Total run time: {:.3f} {}'.format(len(flt_flist), duration, units)
    )
    LOG.info(msg)


    output = [
            fout_long,
            fout_long.replace('.fits','.tra'),
            fout_long.replace('_crj.fits','_spt.fits')
        ]
    LOG.info('Removing generated files from {}'.format(directory))
    for f in output:
        os.system('rm -v {}'.format(f))


def main():
    args = parser.parse_args()
    combine_data(args.dir)

if __name__ == '__main__':
    main()
