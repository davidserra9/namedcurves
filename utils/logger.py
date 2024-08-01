import logging
import os
import datetime

def prepare_logging():
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dirpath = "log_" + timestamp
    save_path = os.path.join(save_path, log_dirpath)
    os.mkdir(save_path)

    handlers = [logging.FileHandler(os.path.join(save_path, "log.txt")), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=handlers,
                        datefmt='%Y-%m-%d %H:%M:%S')

    return save_path