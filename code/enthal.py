#! /usr/bin/env python

from multiprocessing.pool import ThreadPool
import subprocess
import time
import math

import itertools

"""
Configuration classes for 1dx-1dv Vlasov-Poisson (hybrid linearised) solvers
----------------------------------------------------------------------------
"""

class simu_config_1dx1dv:
  def __init__ ( self , Nx , Nv , Tc , ui, nh , alpha , K , dt0 , Tf, output_dir ):
    self.Nx         = Nx
    self.Nv         = Nv
    self.Tc         = Tc
    self.ui         = ui
    self.nh         = nh
    self.alpha      = alpha
    self.K          = K
    self.dt0        = dt0
    self.Tf         = Tf
    self.output_dir = output_dir

  def write ( self , config_name="config.init" ):
    with open(config_name,'w') as f :
      f.write("\n".join([ "{} {}".format(k,v) for (k,v) in self.__dict__.items() ]))

def compute_Nv(Tc):
  import math
  return math.floor(16.0/(math.sqrt(Tc)/20.0))
def compute_nh(Tc):
  import math
  #return 1. - Tcrhoc/Tc
  return 0.2

def default_1dx1dv(Nx=135 , Nv=128 , Tc=0.1 , ui=3.4 , nh=0.2 , alpha=0.01 , K=0.5 , dt0=0.01, Tf=200, output_dir="result"):
  return {
    'Nx'         : Nx,
    'Nv'         : Nv if (Nv) else compute_Nv(Tc),
    'Tc'         : Tc,
    'ui'         : ui,
    'nh'         : nh,
    'alpha'      : alpha,
    'K'          : K,
    'dt0'        : dt0,
    'Tf'         : Tf,
    'output_dir' : output_dir,
  }

"""
Configuration classes for 1dz-3dv Vlasov-Maxwell (hybrid linearised) solvers
----------------------------------------------------------------------------
"""

class simu_config_1dz3dv:
  def __init__ ( self , Nz , Nvx , Nvy , Nvz , dt0 , Tf , nh , v_par , v_perp , B0 , alpha , K , tol , output_dir ):
    self.Nz         = Nz
    self.Nvx        = Nvx
    self.Nvy        = Nvy
    self.Nvz        = Nvz
    self.dt0        = dt0
    self.Tf         = Tf
    self.nh         = nh
    self.v_par      = v_par
    self.v_perp     = v_perp
    self.B0         = B0
    self.alpha      = alpha
    self.K          = K
    self.tol        = tol
    self.output_dir = output_dir

  def write( self , config_name="config.init" ):
    with open(config_name,'w') as f:
      f.write("\n".join([ "{} {}".format(k,v) for (k,v) in self.__dict__.items() ]))


def default_1dz3dv(Nz=27,Nvx=20,Nvy=20,Nvz=43,dt0=0.05,Tf=200,nh=0.2,v_par=0.2,v_perp=0.6,B0=1,alpha=1e-3,K=2.0,tol=1e-5,output_dir="result"):
  return {
    'Nz'        : Nz,
    'Nvx'       : Nvx,
    'Nvy'       : Nvy,
    'Nvz'       : Nvz,
    'dt0'       : dt0,
    'Tf'        : Tf,
    'nh'        : nh,
    'v_par'     : v_par,
    'v_perp'    : v_perp,
    'B0'        : B0,
    'alpha'     : alpha,
    'K'         : K,
    'tol'       : tol,
    'output_dir': output_dir
  }


def worker(label,simu,param,mrproper=True):
  simu_exe = "mtest/hybrid_vmhllf_{}.out".format(simu)

  """
  make_cmd_none = ["make","mrproper" if (mrproper) else None , simu_exe ]
  make_cmd = filter(None,make_cmd_none)
  print("\033[34;1m>\033[0m "," ".join(make_cmd))
  make = subprocess.Popen(make_cmd)
  make.wait()
  """

  print(u"\033[34;1m>\033[0m {}".format(label))
  print(u"\033[34;1m+\033[0m {}".format(simu))
  print(" ".join([ "{} {}".format(k,v) for (k,v) in param.__dict__.items() ]))
  config_file = "config_{}.init".format(simu)
  param.write( config_file )
  time.sleep(0.5)

  cmd = ["./"+simu_exe, config_file ]
  print(" ".join(cmd))
  p = subprocess.Popen(cmd)
  p.wait()

"""
Configuration and launcher
--------------------------
"""

#########################################

params = {'alpha':1e-5,'nh':0.4,'Nz':27,'Nvx':20,'Nvy':20,'Nvz':41,'Tf':200,'tol':3e-5}
simu_params = {
  "verif":(["p22rk44","t5rk44"],[
      simu_config_1dz3dv(**default_1dz3dv(**params,dt0=0.05,output_dir="chap3/verif")) ,
    ]),
  "max":(["mp22rk44","mt5rk44"],[
      simu_config_1dz3dv(**default_1dz3dv(**params,dt0=0.05,output_dir="chap3/max/dt0p05")) ,
      simu_config_1dz3dv(**default_1dz3dv(**params,dt0=0.1 ,output_dir="chap3/max/dt0p1")) ,
    ]),
  "pade":(["mp21rk33","mp12rk33","mp11rk33","mp22rk33"],[
      simu_config_1dz3dv(**default_1dz3dv(**params,dt0=0.1,output_dir="chap3/pade")) ,
    ]),
  "taylor":(["mt1rk33","mt2rk33","mt3rk33","mt4rk33"],[
      simu_config_1dz3dv(**default_1dz3dv(**params,dt0=0.1,output_dir="chap3/taylor")) ,
    ]),
  "dtn":(["mp22dp43","mt5dp43"],[
      simu_config_1dz3dv(**default_1dz3dv(**params,dt0=0.5,output_dir="chap3/dtn")) ,
    ]),
}
launch_simu = []
for key,(simus,params) in simu_params.items() :
  launch_simu.extend(simus)

mrproper = False
num = 2

print("make "+" ".join(launch_simu))
make_cmd_none = ["make","mrproper" if (mrproper) else None ]+[ "mtest/hybrid_vmhllf_{}.out".format(simu) for simu in launch_simu ]
make_cmd = list(filter(None,make_cmd_none))
print("\033[34;1m>\033[0m "," ".join(make_cmd))
make = subprocess.Popen(make_cmd)
make.wait()

tp = ThreadPool(num)
for key,(simus,params) in simu_params.items() :
  for s,p in itertools.product(simus,params):
    tp.apply_async( worker , (key,s,p,) )

tp.close()
tp.join()


