import os
from os.path import join as os_join

from stefutil import *

from ns_bi_encoder.util.data_path import BASE_PATH, PROJ_DIR, DSET_DIR, PKG_NM, MODEL_DIR


__all__ = ['sconfig', 'u']

sconfig = StefConfig(config_file=os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', 'config.json')).__call__
u = StefUtil(
    base_path=BASE_PATH, project_dir=PROJ_DIR, package_name=PKG_NM, dataset_dir=DSET_DIR, model_dir=MODEL_DIR
)
u.dset_path = os_join(u.base_path, u.proj_dir, u.dset_dir)
