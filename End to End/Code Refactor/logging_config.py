# logging_config.py
import logging

# 1. Set default log level here (change this value as needed)
DEFAULT_LOG_LEVEL = logging.INFO  # Switch to logging.ERROR for errors-only by default

# 2. Configure logger with default level
cpr_logger = logging.getLogger("CPR-Analyzer")
cpr_logger.setLevel(DEFAULT_LOG_LEVEL)

# 3. Create console handler with formatter
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 4. Add handler to logger
cpr_logger.addHandler(console_handler)

# 5. Prevent propagation to root logger
cpr_logger.propagate = False