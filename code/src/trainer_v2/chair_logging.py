import logging.config
import os
import re
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


c_log = None
if c_log is None:
    c_log = logging.getLogger('chair')
    c_log.setLevel(logging.INFO)
    format_str = '%(levelname)s\t%(name)s \t%(asctime)s %(message)s'
    formatter = logging.Formatter(format_str,
                                  datefmt='%m-%d %H:%M:%S',
                                  )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(ch)
    c_log.info("Chair logging init")
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.propagate = False

