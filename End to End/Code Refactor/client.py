import socket
import json
from logging_config import cpr_logger

HOST = 'localhost'  # The server's hostname or IP address
PORT = 5000        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    cpr_logger.info(f"Connected to {HOST}:{PORT}")
    
    try:
        while True:
            data = s.recv(1024)
            if not data:
                break
                
            # Split messages (in case multiple JSONs in buffer)
            for line in data.decode('utf-8').split('\n'):
                if line.strip():
                    try:
                        warnings = json.loads(line)
                        cpr_logger.info("\nReceived warnings:")
                        cpr_logger.info(f"Status: {warnings['status']}")
                        cpr_logger.info(f"Posture Warnings: {warnings['posture_warnings']}")
                        cpr_logger.info(f"Rate/Depth Warnings: {warnings['rate_and_depth_warnings']}")
                    except json.JSONDecodeError:
                        cpr_logger.info("Received invalid JSON")
    except KeyboardInterrupt:
        cpr_logger.info("Disconnecting...")