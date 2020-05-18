# Verbosity level of logging:
##############
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
verbosity_level = 'INFO'

# Some common useful packages
from contextlib import contextmanager
from os import getcwd as pwd
from os.path import join
from sys import argv, path
import argparse
from datetime import datetime
import glob
import logging
import numpy as np
import os
import sys
import time
import math
import signal

def get_logger(verbosity_level, use_error_log=False):
    """Set logging format to something like:
        2019-04-25 12:52:51,924 INFO score.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger

logger = get_logger(verbosity_level)


def mprint(msg):
    """info"""
    cur_time = datetime.now().strftime('%m-%d %H:%M:%S')
    logger.info("[{}] {}".format(cur_time, msg))


class TimeoutException(Exception):
    pass

class Timer:
    def __init__(self):
        self.duration = 0
        self.total = None
        self.remain = None
        self.exec = None

    def set(self, time_budget):
        self.total = time_budget
        self.remain = time_budget
        self.exec = 0

    @contextmanager
    def time_limit(self, pname):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(int(math.ceil(self.remain)))
        start_time = time.time()

        try:
            yield
        finally:
            exec_time = time.time() - start_time
            signal.alarm(0)
            self.exec += exec_time
            self.duration += exec_time
            self.remain = self.total - self.exec

        mprint('{} success, time spent so far {} sec'.format(pname, self.exec))

        if self.remain <= 0:
            raise TimeoutException("Timed out!")


def _HERE(*args):
    """Helper function for getting the current directory of this script."""
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(h, *args))

def write_start_file(output_dir, start_time=None, time_budget=None,
                     task_name=None):
    """Create start file 'start.txt' in `output_dir` with ingestion's pid and
    start time.

    The content of this file will be similar to:
        ingestion_pid: 1
        task_name: beatriz
        time_budget: 2400.0
        start_time: 1557923830.3012087
        0: 7.116559
        1: 14.112311
        <more timestamps of predictions>
    """
    ingestion_pid = os.getpid()
    start_filename =  'start.txt'
    start_filepath = os.path.join(output_dir, start_filename)
    with open(start_filepath, 'w') as f:
        f.write('ingestion_pid: {}\n'.format(ingestion_pid))
        f.write('task_name: {}\n'.format(task_name))
        f.write('time_budget: {}\n'.format(time_budget))
        f.write('start_time: {}\n'.format(start_time))
    logger.debug("Finished writing 'start.txt' file.")

def write_timestamp(output_dir, predict_idx, timestamp):
    start_filename =  'start.txt'
    start_filepath = os.path.join(output_dir, start_filename)
    with open(start_filepath, 'a') as f:
        f.write('{}: {}\n'.format(predict_idx, timestamp))
    logger.debug("Wrote timestamp {} to 'start.txt' for predition {}."\
                .format(timestamp, predict_idx))

class ModelApiError(Exception):
    pass

class BadPredictionShapeError(Exception):
    pass

def _parse_args():
    # Parse directories from input arguments
    root_dir = _HERE(os.pardir)
    default_dataset_dir = join(root_dir, "sample_data")
    default_output_dir = join(root_dir, "sample_result_submission")
    default_ingestion_program_dir = join(root_dir, "ingestion_program")
    default_code_dir = join(root_dir, "sample_code_submission")
    default_score_dir = join(root_dir, "scoring_output")
    default_time_budget = 1200
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default=default_dataset_dir,
                        help="Directory storing the dataset (containing " +
                             "e.g. adult.data/)")
    parser.add_argument('--output_dir', type=str,
                        default=default_output_dir,
                        help="Directory storing the predictions. It will " +
                             "contain e.g. [start.txt, adult.predict_0, " +
                             "adult.predict_1, ..., end.txt] when ingestion " +
                             "terminates.")
    parser.add_argument('--ingestion_program_dir', type=str,
                        default=default_ingestion_program_dir,
                        help="Directory storing the ingestion program " +
                             "`ingestion.py` and other necessary packages.")
    parser.add_argument('--code_dir', type=str,
                        default=default_code_dir,
                        help="Directory storing the submission code " +
                             "`model.py` and other necessary packages.")
    parser.add_argument('--score_dir', type=str,
                        default=default_score_dir,
                        help="Directory storing the scoring output " +
                             "e.g. `scores.txt` and `detailed_results.html`.")
    parser.add_argument("--time_budget", type=float,
                        default=default_time_budget,
                        help="Time budget for running model if not specified "
                             "in meta.json.")
    args = parser.parse_args()
    logger.debug("Parsed args are: " + str(args))
    logger.debug("-" * 50)
    return args

def _prepare_work_dir(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.score_dir, exist_ok=True)

# =========================== BEGIN PROGRAM ================================

def _main(args):
    # Mark starting time of ingestion
    start = time.time()
    logger.info("="*5 + " Start ingestion program. ")

    #### Check whether everything went well
    ingestion_success = True

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    ingestion_program_dir= args.ingestion_program_dir
    code_dir= args.code_dir
    score_dir = args.score_dir
    time_budget = args.time_budget

    if dataset_dir.endswith('run/input') and\
        code_dir.endswith('run/program'):
        logger.debug("Since dataset_dir ends with 'run/input' and code_dir "
                    "ends with 'run/program', suppose running on " +
                    "CodaLab platform. Modify dataset_dir to 'run/input_data' "
                    "and code_dir to 'run/submission'. " +
                    "Directory parsing should be more flexible in the code of " +
                    "compute worker: we need explicit directories for " +
                    "dataset_dir and code_dir.")
        dataset_dir = dataset_dir.replace('run/input', 'run/input_data')
        code_dir = code_dir.replace('run/program', 'run/submission')

    # Show directories for debugging
    logger.debug("sys.argv = " + str(sys.argv))
    logger.debug("Using dataset_dir: " + dataset_dir)
    logger.debug("Using output_dir: " + output_dir)
    logger.debug("Using ingestion_program_dir: " + ingestion_program_dir)
    logger.debug("Using code_dir: " + code_dir)

    # Our libraries
    path.append(ingestion_program_dir)
    path.append(code_dir)
    #IG: to allow submitting the starting kit as sample submission
    path.append(code_dir + '/sample_code_submission')
    import data_io
    from dataset import AutoSpeechDataset # THE class of AutoNLP datasets

    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(dataset_dir)
    #### Delete zip files and metadata file
    datanames = [x for x in datanames if x.endswith('.data')]

    if len(datanames) != 1:
        raise ValueError("{} datasets found in dataset_dir={}!\n"\
                        .format(len(datanames), dataset_dir) +
                        "Please put only ONE dataset under dataset_dir.")

    basename = datanames[0]
    D = AutoSpeechDataset(os.path.join(dataset_dir, basename))
    metadata = D.get_metadata()
    time_budget = metadata.get("time_budget", time_budget)
    logger.info("Time budget: {}".format(time_budget))

    write_start_file(output_dir, start_time=start, time_budget=time_budget,
                     task_name=basename.split('.')[0])

    logger.info("************************************************")
    logger.info("******** Processing dataset " + basename[:-5].capitalize() +
                " ********")
    logger.info("************************************************")

    ##### Begin creating training set and test set #####
    logger.info("Reading training set and test set...")
    D.read_dataset()
    ##### End creating training set and test set #####

    ## Get correct prediction shape
    num_examples_test = D.get_test_num()
    output_dim = D.get_class_num()
    correct_prediction_shape = (num_examples_test, output_dim)

    try:
        # ========= Creating a model
        timer = Timer()
        timer.set(20 * 60) # 20 min for participants to initializing and install other packages
        with timer.time_limit("Importing model"):
            from model import Model # in participants' model.py

        ##### Begin creating model #####
        logger.info("Creating model...")
        with timer.time_limit('Initialization'):
            M = Model(metadata)
        ###### End creating model ######
    except TimeoutException as e:
        logger.info("[-] Initialization phase exceeded time budget. Move to train/predict phase")
    except Exception as e:
        logger.info("Failed to initializing model.")
        logger.error("Encountered exception:\n" + str(e), exc_info=True)
        raise
    finally:
        try:
            timer = Timer()
            timer.set(time_budget)
            # Check if the model has methods `train` and `test`.
            for attr in ['train', 'test']:
                if not hasattr(M, attr):
                    raise ModelApiError("Your model object doesn't have the method " +
                                        "`{}`. Please implement it in model.py.")

            # Check if model.py uses new done_training API instead of marking
            # stopping by returning None
            use_done_training_api = hasattr(M, 'done_training')
            if not use_done_training_api:
                logger.warning("Your model object doesn't have an attribute " +
                            "`done_training`. But this is necessary for ingestion " +
                            "program to know whether the model has done training " +
                            "and to decide whether to proceed more training. " +
                            "Please add this attribute to your model.")

            # Keeping track of how many predictions are made
            prediction_order_number = 0

            # Start the CORE PART: train/predict process
            while (not (use_done_training_api and M.done_training)):

                # Train the model
                logger.info("Begin training the model...")
                remaining_time_budget = timer.remain
                with timer.time_limit('training'):
                    M.train(D.get_train(), remaining_time_budget=timer.remain)
                logger.info("Finished training the model.")

                # Make predictions using the trained model
                logger.info("Begin testing the model by making predictions " +
                            "on test set...")
                remaining_time_budget = timer.remain
                with timer.time_limit('predicting'):
                    Y_pred = M.test(D.get_test(),
                                    remaining_time_budget=remaining_time_budget)
                logger.info("Finished making predictions.")

                if Y_pred is None: # Stop train/predict process if Y_pred is None
                    logger.info("The method model.test returned `None`. " +
                                "Stop train/predict process.")
                    break
                else: # Check if the prediction has good shape
                    prediction_shape = tuple(Y_pred.shape)
                    if prediction_shape != correct_prediction_shape:
                        raise BadPredictionShapeError(
                            "Bad prediction shape! Expected {} but got {}."\
                            .format(correct_prediction_shape, prediction_shape)
                        )
                # Write timestamp to 'start.txt'
                write_timestamp(output_dir, predict_idx=prediction_order_number,
                                timestamp=timer.exec)
                # Prediction files: adult.predict_0, adult.predict_1, ...
                filename_test = basename[:-5] + '.predict_' +\
                    str(prediction_order_number)
                # Write predictions to output_dir
                tmp_pred = np.argmax(Y_pred,axis=1)
                # data_io.write(os.path.join(output_dir,filename_test), Y_pred)
                data_io.write(os.path.join(output_dir,filename_test), tmp_pred)
                prediction_order_number += 1
                logger.info("[+] {0:d} predictions made, time spent so far {1:.2f} sec"\
                            .format(prediction_order_number, time.time() - start))
                logger.info("[+] Time left {0:.2f} sec".format(timer.remain))
        except TimeoutException as e:
            logger.info("[-] Ingestion program exceeded time budget. Predictions "
                        "made so far will be used for evaluation.")
        except Exception as e:
            ingestion_success = False
            logger.info("Failed to run ingestion.")
            logger.error("Encountered exception:\n" + str(e), exc_info=True)
            raise
        finally:
            # Finishing ingestion program
            end_time = time.time()
            overall_time_spent = end_time - start

            # Write overall_time_spent to a end.txt file
            end_filename = 'end.txt'
            with open(os.path.join(output_dir, end_filename), 'w') as f:
                f.write('ingestion_duration: ' + str(overall_time_spent) + '\n')
                f.write('ingestion_success: ' + str(int(ingestion_success)) + '\n')
                f.write('end_time: ' + str(end_time) + '\n')
                logger.info("Wrote the file {} marking the end of ingestion."\
                            .format(end_filename))
                if ingestion_success:
                    logger.info("[+] Done. Ingestion program successfully terminated.")
                    logger.info("[+] Overall time spent %5.2f sec " % overall_time_spent)
                else:
                    logger.info("[-] Done, but encountered some errors during ingestion.")
                    logger.info("[-] Overall time spent %5.2f sec " % overall_time_spent)

            # Copy all files in output_dir to score_dir
            os.system("cp -R {} {}".format(os.path.join(output_dir, '*'), score_dir))
            logger.debug("Copied all ingestion output to scoring output directory.")

            logger.info("[Ingestion terminated]")


if __name__=="__main__":
    import traceback
    args = _parse_args()
    err_log_file = os.path.join(args.score_dir, 'ingestion_error.log')
    try:
        _prepare_work_dir(args)
        _main(args)
    except Exception:
        errmsg = '\n'.join(traceback.format_exc().split('\n')[:-2])
        with open(err_log_file, 'w') as fh:
            fh.write(errmsg)
        time.sleep(3)
        raise

