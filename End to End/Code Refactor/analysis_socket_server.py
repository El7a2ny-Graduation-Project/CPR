import socket
import json
from threading import Thread
from queue import Queue
import threading
from logging_config import cpr_logger
import queue

class AnalysisSocketServer:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None
        self.running = False
        self.warning_queue = Queue()
        self.connection_event = threading.Event()
        cpr_logger.info(f"[SOCKET] Server initialized on {host}:{port}")

    def start_server(self):
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        self.running = True
        Thread(target=self._accept_connections, daemon=True).start()

    def _accept_connections(self):
        while self.running:
            try:
                self.conn, addr = self.sock.accept()
                cpr_logger.info(f"[SOCKET] Connected by {addr}")
                self.connection_event.set()  # Signal that connection was made
                Thread(target=self._handle_client, args=(self.conn,), daemon=True).start()
            except Exception as e:
                cpr_logger.info(f"[SOCKET] Connection error: {str(e)}")

    def wait_for_connection(self, timeout=None):
        """Block until a client connects"""
        cpr_logger.info("[SOCKET] Waiting for client connection...")
        self.connection_event.clear()  # Reset the event
        return self.connection_event.wait(timeout)

    def _handle_client(self, conn):
        while self.running:
            try:
                # Block until a warning is available (reduces CPU usage)
                warnings = self.warning_queue.get(block=True, timeout=0.1)
                serialized = json.dumps(warnings) + "\n"
                conn.sendall(serialized.encode('utf-8'))
            except queue.Empty:
                continue  # Timeout allows checking self.running periodically
            except (BrokenPipeError, ConnectionResetError):
                cpr_logger.info("[SOCKET] Client disconnected")
                break
            except Exception as e:
                cpr_logger.info(f"[SOCKET] Error: {str(e)}")
                break
        conn.close()

    def stop_server(self):
        self.running = False
        self.sock.close()
        cpr_logger.info("[SOCKET] Server stopped")