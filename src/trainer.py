import subprocess
import os
import signal
import pandas as pd
import time

class Trainer:
    def __init__(self, script_path="train.py", log_path="training_log.csv"):
        self.script_path = script_path
        self.log_path = log_path
        self.process = None
        
    def start_training(self, epochs=50, batch_size=32, lr=0.001):
        if self.is_running():
            return False
            
        # Clear old log
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
            
        # Construct command (passing args if train.py supported them, 
        # for now we assume train.py reads config or defaults, 
        # but let's pass them as env vars or args if we modify train.py.
        # For this iteration, we will just run it as is, but we *should* modify train.py to accept args.
        # Let's assume we will modify train.py next.)
        cmd = ["python", self.script_path, "--epochs", str(epochs), "--batch_size", str(batch_size), "--lr", str(lr)]
        
        self.process = subprocess.Popen(cmd, cwd=os.getcwd())
        return True
        
    def stop_training(self):
        if self.process:
            self.process.terminate()
            self.process = None
            return True
        return False
        
    def is_running(self):
        if self.process:
            if self.process.poll() is None:
                return True
            else:
                self.process = None # Clean up if finished
        return False
        
    def get_logs(self):
        if os.path.exists(self.log_path):
            try:
                return pd.read_csv(self.log_path)
            except:
                return pd.DataFrame()
        return pd.DataFrame()
