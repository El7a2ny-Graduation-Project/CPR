import logging
import logging.handlers
import queue

# 1. Configure once
log_queue = logging.handlers.QueueHandler(queue.Queue(-1))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 2. Single shared instance 
cpr_logger = logging.getLogger("CPR-Analyzer")