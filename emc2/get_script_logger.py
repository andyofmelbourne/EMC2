import logging
from pathlib import Path


def get_logger(
    filemode='a',
    working_directory='',
    name='emc2'
):
    # prevent these modules from flooding my log in debug mode
    logging.getLogger('pytools').setLevel(logging.WARNING)
    logging.getLogger('pyopencl').setLevel(logging.WARNING)

    fnam = Path(working_directory).joinpath(f'{name}.log')
    logger = logging.getLogger(name)
    # format_str = 'epoch:level:pid:filename:function_name:lineno: msg'
    FORMAT = "%(created)f:%(levelname)s:%(process)d:%(filename)s"\
             ":%(funcName)s:%(lineno)d: %(message)s"
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG,
        filename=fnam,
        filemode=filemode
    )
    # logging.info(format_str)
    return logger
