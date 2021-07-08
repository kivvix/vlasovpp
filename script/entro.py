#! /usr/bin/env python

from multiprocessing.pool import ThreadPool
import subprocess

def params ( s ):
  cmd = []
  b = 'm' in s
  if b :
    cmd.append('-m')
  meth = s[int(b)]
  cmd.append("-{}".format(meth))
  if meth == 'p' or meth == 't' :
    n = int(s[int(b)+1])
    m = " "
    if meth == 'p':
      try:
        m = int(s[int(b)+2])
      except ValueError:
        m = n
    cmd.extend([str(n),str(m)])
  rk = s[-4:]
  cmd.append("--lrk={}".format(rk.upper()))

  return cmd

def worker ( s ):
  cmd = ["./code_gen3.py"]
  cmd.extend(params(s))
  cmd.extend("-o ../code/mtest".split())
  command_line = " ".join(cmd)
  print(command_line)
  #p = subprocess.Popen(command_line.split())
  #p.wait()

num = 1
tp = ThreadPool(num)
#for s in "mp22dp43 mt5dp43".split():
for s in "erk44 p22rk44 t5rk44 mp22rk44 mt5rk44 mp21rk33 mp12rk33 mp11rk33 mp22rk33 mt1rk33 mt2rk33 mt3rk33 mt4rk33 mp22dp43 mt5dp43".split():
  tp.apply_async( worker , (s,) )

tp.close()
tp.join()


