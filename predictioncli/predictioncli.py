#!/usr/bin/env python-real

import os
import sys
import json
 
def main(surf, out, rot, res, model, scal):
  # Opening JSON file
  with open('env.json') as json_file:
      env = json.load(json_file)

  fileDir = os.path.dirname(os.path.abspath(__file__))
  os.chdir(fileDir)
  os.chdir('../seg_code/FiboSeg/')

  #command_to_execute = ["python","DEMO.py"]
  command_to_execute = ["python", "predict_v3.py","--surf", surf, "--out", out,"--rot",rot ,"--res",res,"--model",model,"--scal",scal]
  print('command: ', command_to_execute)
  from subprocess import check_output
  check_output(command_to_execute,env=env)
  print('done')

if __name__ == "__main__":
  if len (sys.argv) < 7:
    print("Usage: predictioncli <surf> <out> <rot> <res> <model> <scal>")
    sys.exit (1)
  main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],sys.argv[6])
