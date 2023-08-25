from multiprocessing import Pool
from typing import Union, Tuple

from predictor.utilities.file_and_folder_operations import *


def convert_trainer_plans_config_to_identifier(trainer_name, plans_identifier, configuration):
    return f'{trainer_name}__{plans_identifier}__{configuration}'


def convert_identifier_to_trainer_plans_config(identifier: str):
    return os.path.basename(identifier).split('__')



def get_ensemble_name(model1_folder, model2_folder, folds: Tuple[int, ...]):
    identifier = 'ensemble___' + os.path.basename(model1_folder) + '___' + \
                 os.path.basename(model2_folder) + '___' + folds_tuple_to_string(folds)
    return identifier


def convert_ensemble_folder_to_model_identifiers_and_folds(ensemble_folder: str):
    prefix, *models, folds = os.path.basename(ensemble_folder).split('___')
    return models, folds


def folds_tuple_to_string(folds: Union[List[int], Tuple[int, ...]]):
    s = str(folds[0])
    for f in folds[1:]:
        s += f"_{f}"
    return s


def folds_string_to_tuple(folds_string: str):
    folds = folds_string.split('_')
    res = []
    for f in folds:
        try:
            res.append(int(f))
        except ValueError:
            res.append(f)
    return res


def check_workers_alive_and_busy(export_pool: Pool, worker_list: List, results_list: List, allowed_num_queued: int = 0):
    """

    returns True if the number of results that are not ready is greater than the number of available workers + allowed_num_queued
    """
    alive = [i.is_alive() for i in worker_list]
    if not all(alive):
        raise RuntimeError('Some background workers are no longer alive')

    not_ready = [not i.ready() for i in results_list]
    if sum(not_ready) >= (len(export_pool._pool) + allowed_num_queued):
        return True
    return False
