import logging


logging.basicConfig(
    filename="app.log", 
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%S.%s+0800',
)

logger = logging.getLogger()

