import logging
from pathlib import Path


def get_logger(
    filemode='a',
    working_directory='',
    name='emc2'
):
    fnam = Path(working_directory).joinpath(f'{name}.log')
    logger = logging.getLogger(name)
    # format_str = 'epoch:level:pid:filename:function_name:lineno: msg'
    FORMAT = "%(created)f:%(levelname)s:%(process)d:%(filename)s"\
             ":%(funcName)s:%(lineno)d: %(message)s"
    logging.basicConfig(
        format=FORMAT,
        level=logging.INFO,
        filename=fnam,
        filemode=filemode
    )
    # logging.info(format_str)
    return logger
