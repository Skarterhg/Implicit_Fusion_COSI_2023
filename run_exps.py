import subprocess
import os
import time
import numpy as np
import tqdm
import argparse


logs_folder = "logs"
if not os.path.exists(logs_folder):
  os.makedirs(logs_folder)

parser = argparse.ArgumentParser()
parser.add_argument("--cover", type=bool, default=False, help="SNR of the experiment")
config = parser.parse_args()


cover = config.cover


if cover:
   print("Runing cover")
   experiments = np.linspace(0, 0.2, 5)
else:
   print("Runing snr")
   experiments = [1,5, 10,15, 20]

num_batch_trainings = 2

### Divide experiments in num_batch_trainings

#batches = np.array_split(np.arange(len(experiments)), np.ceil(len(experiments)/num_batch_trainings))

ids_exps = np.arange(len(experiments))

processes = {}
running_exps = []
started_exps = []
ended_exps = []
idex = 0

pbar = tqdm.tqdm(total=len(ids_exps))
while len(ended_exps)<len(ids_exps):
   
   if len(running_exps) < num_batch_trainings and idex < len(ids_exps):
      snr = experiments[idex]
      #print("Opening proceedure for wz = " + str(wz) + " and wc = " + str(wc))
      output_file_1 =  open(os.path.join(logs_folder, f"log_script_system{snr}.txt"), "w")
      error_file_1 =  open(os.path.join(logs_folder, f"err_script_system{snr}.txt"), "w")
      print(snr)
      if cover:
            processes[idex] = subprocess.Popen(["./run_exp.sh", "cover", str(snr)], stdout=output_file_1, stderr=error_file_1, bufsize=1)
      else:
            processes[idex] = subprocess.Popen(["./run_exp.sh", "snr", str(snr)], stdout=output_file_1, stderr=error_file_1, bufsize=1)
      running_exps.append(idex)

      
      idex += 1

   else:
      pbar.set_description(f"Running {running_exps} experiments")
      for i, pid in enumerate(running_exps):
         if processes[pid].poll() is not None:
            #print(f"process {pid} is done")
            pbar.update(1)
            processes[pid].kill()
            del processes[pid]
            del running_exps[i]
            ended_exps.append(pid)
            
         else:
            #print(f"process {pid} is running")
            time.sleep(3)
pbar.close()

