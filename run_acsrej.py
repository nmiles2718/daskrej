#!/usr/bin/env python

import argparse
from acstools import acsrej
from astropy.io import fits
import glob
import numpy as np
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('-dir', type=str,
                    help='/path/to/directory/to/process/')



_OUTPUT_DIR = os.path.join(
    '/',
    *os.path.dirname(os.path.abspath(__file__)).split('/'),
    'acsrej_output'
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


def copy_updated_flts(flist):
    print('Copying data products to output directory')
    for f in flist:
        os.system('cp {} {}/{}'.format(f,
                                          _OUTPUT_DIR,
                                          os.path.basename(f)))




def combine_data(directory):
    st = time.time()
    print('processing data from {}'.format(directory))
    flt_list = glob.glob(directory+'/*flt.fits')
    fout_long = directory+'/acsrej_crj.fits'
    run_acsrej(flt_list,
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
        'Total run time: {:.3f} {}'.format(len(flt_list), duration, units)
    )
    print(msg)


    copy_updated_flts(flt_list)
    output = [
            fout_long,
            fout_long.replace('.fits','.tra'),
            fout_long.replace('_crj.fits','_spt.fits')
        ]
    print('Removing generated files from {}'.format(directory))
    for f in output:
        os.system('rm -v {}'.format(f))

def main():
    args = parser.parse_args()
    combine_data(args.dir)

if __name__ == '__main__':
    main()
