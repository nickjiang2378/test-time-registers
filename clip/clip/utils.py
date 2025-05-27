import time
import torch
from datetime import datetime
import os

class SaveOcassionally:
  def __init__(self, out, every_sec = None, every_count = None):
    assert every_sec != None or every_count != None

    self.out = out
    self.curr_time = time.time()
    self.every_sec = every_sec
    self.cnt = 0
    self.every_count = every_count
    self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if "TIMESTAMP" in self.out:
      self.out = self.out.replace("TIMESTAMP", self.timestamp)
      print(f"Replacing TIMESTAMP with {self.timestamp}")

    # Ensure the directory exists
    out_dir = os.path.abspath(os.path.dirname(self.out))
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

  def save(self, obj):
    self.cnt += 1
    if self.every_sec != None and time.time() - self.curr_time > self.every_sec:
      torch.save(obj, self.out)
    elif self.every_count != None and self.cnt % self.every_count == 0:
      torch.save(obj, self.out)

  def force_save(self, obj):
    torch.save(obj, self.out)
