#!/usr/bin/env python3
import subprocess
import time

Tcrhoc = 0.08

class simu_config_1dx1dv:
  def __init__ ( self , Nx , Nv , Tc , Tf , alpha , output_dir ):
    self.Nx = Nx
    self.Nv = Nv
    self.Tc = Tc
    self.Tf = Tf
    self.alpha = alpha
    self.output_dir = output_dir

  def write ( self , config_name="config.init" ):
    with open(config_name,'w') as f :
      f.write("\n".join([ "{} {}".format(k,v) for (k,v) in self.__dict__.items() ]))

class simu_config_1dz3dv:
  def __init__ ( self , Nz , Nvx , Nvy , Nvz , dt0 , Tf , nh , v_par , v_perp , alpha , K , output_dir ):
    self.Nz = Nz
    self.Nvx = Nvx
    self.Nvy = Nvy
    self.Nvz = Nvz
    self.dt0 = dt0
    self.Tf  = Tf
    self.nh  = nh
    self.v_par  = v_par
    self.v_perp = v_perp
    self.alpha = alpha
    self.K = K
    self.output_dir = output_dir

  def write( self , config_name="config.init" ):
    with open(config_name,'w') as f:
      f.write("\n".join([ "{} {}".format(k,v) for (k,v) in self.__dict__.items() ]))

def compute_Nv(Tc):
  import math
  return math.floor(16.0/(math.sqrt(Tc)/20.0))
def compute_alpha(Tc):
  import math
  #return 1. - Tcrhoc/Tc
  return 0.2

def dic_init(Tc,d):
  return {'Tc':Tc,'Nv':compute_Nv(Tc),'Nx':135,'alpha':compute_alpha(Tc),'Tf':10.0,'output_dir':d}

configs = [
  simu_config_1dx1dv(**dic_init(0.08 ,"compare/alpha2/tm0p08"))  ,
  simu_config_1dx1dv(**dic_init(0.1  ,"compare/alpha2/tm0p1"))   ,
  simu_config_1dx1dv(**dic_init(0.125,"compare/alpha2/tm0p125")) ,
  simu_config_1dx1dv(**dic_init(0.15 ,"compare/alpha2/tm0p15"))  ,
  simu_config_1dx1dv(**dic_init(0.175,"compare/alpha2/tm0p175")) ,
  simu_config_1dx1dv(**dic_init(0.2  ,"compare/alpha2/tm0p2"))
]

"""
configs.append(simu_config(Tc=1e-3,Nv=10120,Nx=135,Tf=10.0,output_dir="compare/tm3"))
configs.append(simu_config(Tc=1e-2,Nv=3200 ,Nx=135,Tf=10.0,output_dir="compare/tm2"))
configs.append(simu_config(Tc=1e-1,Nv=1012 ,Nx=135,Tf=10.0,output_dir="compare/tm1"))
configs.append(simu_config(Tc=1e-4,Nv=32000,Nx=135,Tf=10.0,output_dir="compare/tm4"))
"""

#for c in reversed(configs) :
#  print(">--")
#  print(" ".join([ "{} {}".format(k,v) for (k,v) in c.__dict__.items() ]))
#  c.write("config.init")
#  subprocess.run("./cmp_tb.out   config.init".split(),shell=True,check=True)
#  subprocess.run("./cmp_vhll.out config.init".split(),shell=True,check=True)

def default_1dz3dv(Nz=27,Nvx=56,Nvy=56,Nvz=57,dt0=0.05,Tf=200,nh=0.2,v_par=0.2,v_perp=0.6,alpha=1e-4,K=2.0,output_dir="result"):
  return {
    'Nz' : Nz,
    'Nvx': Nvx,
    'Nvy': Nvy,
    'Nvz': Nvz,
    'dt0': dt0,
    'Tf' : Tf,
    'nh' : nh,
    'v_par' : v_par,
    'v_perp': v_perp,
    'alpha' : alpha,
    'K': K,
    'output_dir': output_dir
  }

configs = [
          #simu_config_1dz3dv(**default_1dz3dv(v_par=0.2,v_perp=0.6,nh=0.2,alpha=1e-4,output_dir="runs/test1.1")),
          #simu_config_1dz3dv(**default_1dz3dv(v_par=0.2,v_perp=0.6,nh=0.1,alpha=1e-4,output_dir="runs/test1.2")),
          #simu_config_1dz3dv(**default_1dz3dv(v_par=0.2,v_perp=0.53,nh=0.24,alpha=1e-4,output_dir="runs/test2")),
          #simu_config_1dz3dv(**default_1dz3dv(dt0=0.01,K=2.0,output_dir="runs/test3.1")),
          #simu_config_1dz3dv(**default_1dz3dv(dt0=0.01,K=3.0,output_dir="runs/test3.2")),
          #simu_config_1dz3dv(**default_1dz3dv(dt0=0.01,K=1.5,output_dir="runs/test3.3")),
          #simu_config_1dz3dv(**default_1dz3dv(dt0=0.05,Nz=15,Nvx=20,Nvy=20,Nvz=21,v_par=0.2,v_perp=0.6,nh=0.2,alpha=1e-4,output_dir="runs/test13.1")),
          simu_config_1dz3dv(**default_1dz3dv(dt0=0.005,Nz=27,Nvx=32,Nvy=32,Nvz=33,v_par=0.2,v_perp=0.6,nh=0.4,alpha=1e-4,output_dir="runs/B0_0p1")),
        ]
simus = ["hybrid1dx3dv.out","hybrid1dx3dv_lawson.out","hybrid1dx3dv_lawson_filtre.out"]
process = []
# be sure to compile simulation project
print("\033[2J")
make = subprocess.Popen(["make","mrproper"]+simus)
make.wait()
# and launch all simulations in configs
print(u"\033[7;49;34m** launch simulations 🚀 **\033[0m")
for c in configs:
  print(">--")
  print(" ".join([ "{} {}".format(k,v) for (k,v) in c.__dict__.items() ]))
  c.write("config.init")
  for i,simu in enumerate(simus):
    process.append(subprocess.Popen(["./"+simu,"config.init",25+i],shell=True))
    time.sleep(0.5)
  time.sleep(1.0)

print("\033["+str(25+len(simus)+2)+";H0 ...wait process...")
for p in process:
  p.wait()
print("\033["+str(25+len(simus)+3)+u";H0\033[7;49;32m** finish **\033[0m\n")
