#! /usr/bin/env python3

import sympy as sp
import dataclasses

import cgeneration as cg

r"""
# Lawson scheme

In this module we present some Lawson methods. Each function
represents a Lawson method, and returns an object that contains every
stages of the method.
"""

@dataclasses.dataclass
class lawson:
  stages = []
  is_embeded:bool = False
  un = None

  def __init__(self,computed_stages,expr_stages,dt_stages,is_embeded,un=None):
    for _lhs,_rhs,dt in zip(computed_stages,expr_stages,dt_stages):
      lhs,rhs = (cg.Uhs(),cg.Uhs())
      lhs.dt = dt
      lhs[:] = [ str(ui) for ui in _lhs ]
      rhs[:] = _rhs

      self.stages.append( (lhs,rhs) )
    self.is_embeded = is_embeded
    if is_embeded:
      self.un = cg.Uhs()
      self.un[:] = [ str(ui) for ui in un ]

def LRK11 ( t , dt , eLt , N ):
  Un, = [ cg.vector_stage_idx(sname) for sname in "".split(',') ]
  print("+ stage 1")
  stage_Un1 = eLt.subs(t,dt)*Un + dt/2*eLt.subs(t,dt)*N(Un)

  expr_stages = [stage_Un1]
  computed_stages = [ cg.vector_stage(sname) for sname in "".split(",") ]
  dt_stages = [ 0. ]

  return lawson( computed_stages , expr_stages , dt_stages , is_embeded=False )

def LRK44 ( t , dt , eLt , N ):
  Un , U1 , U2 , U3 = [ cg.vector_stage_idx(sname) for sname in ",1,2,3".split(',') ]
  print("+ stage 1")
  stage_U1  =  eLt.subs(t,dt/2)*Un + dt/2*eLt.subs(t,dt/2)*N(Un)
  print("+ stage 2")
  stage_U2  =  eLt.subs(t,dt/2)*Un + dt/2*N(U1)
  print("+ stage 3")
  stage_U3  =  eLt.subs(t,dt)*Un   + dt*eLt.subs(t,dt/2)*N(U2)
  print("+ stage n+1")
  stage_Un1 = -eLt.subs(t,dt)*Un/3 + eLt.subs(t,dt/2)*U1/3 + 2*eLt.subs(t,dt/2)*U2/3 + U3/3 + dt/6*N(U3)

  expr_stages = [stage_U1,stage_U2,stage_U3,stage_Un1]
  computed_stages = [ cg.vector_stage(sname) for sname in "1,2,3,".split(",") ]
  dt_stages = [ 0. , 0.5 , 0.5 , 1.0 ]

  return lawson( computed_stages , expr_stages , dt_stages , is_embeded=False )

def LDP43 ( t , dt , eLt , N ):
  Un , U1 , U2 , U3 , U4 = [ cg.vector_stage_idx(sname) for sname in ",1,2,3,4".split(',') ]
  print("+ stage 1")
  stage_U1 =  eLt.subs(t,dt/2)*Un + dt/2*eLt.subs(t,dt/2)*N(Un)
  print("+ stage 2")
  stage_U2 =  eLt.subs(t,dt/2)*Un + dt/2*N(U1)
  print("+ stage 3")
  stage_U3 =  eLt.subs(t,dt)*Un   + dt*eLt.subs(t,dt/2)*N(U2)
  print("+ stage 4")
  stage_U4 = -eLt.subs(t,dt)*Un/3 + eLt.subs(t,dt/2)*U1/3 + 2*eLt.subs(t,dt/2)*U2/3 + U3/3 + dt/6*N(U3)
  print("+ stage 5")
  stage_U5 = -eLt.subs(t,dt)*Un/5 + eLt.subs(t,dt/2)*U1/5 + 2*eLt.subs(t,dt/2)*U2/5 + U3/5 + 2*U4/5 + dt/10*N(U4)

  expr_stages = [stage_U1,stage_U2,stage_U3,stage_U4,stage_U5]
  computed_stages = [ cg.vector_stage(sname) for sname in "1,2,3,4,5".split(",") ]
  dt_stages = [ 0. , 0.5 , 0.5 , 1.0 , 1.0 ]

  return lawson( computed_stages , expr_stages , dt_stages , is_embeded=True , un=cg.vector_stage("") )

def LRK33 ( t , dt , eLt , N ):
  Un , U1 , U2 = [ cg.vector_stage_idx(sname) for sname in ",1,2".split(',') ]
  print("+ stage 1")
  stage_U1  =  eLt.subs(t,dt)*Un + dt*eLt.subs(t,dt)*N(Un)
  print("+ stage 2")
  stage_U2  =  0.75*eLt.subs(t,dt/2)*Un + 0.25*eLt.subs(t,-dt/2)*U1 + 0.25*dt*eLt.subs(t,-dt/2)*N(U1)
  print("+ stage n+1")
  stage_Un1 =  eLt.subs(t,dt)*Un/3 + 2*eLt.subs(t,dt/2)*U2/3 + 2*dt/3*eLt.subs(t,dt/2)*N(U2)

  expr_stages = [stage_U1,stage_U2,stage_Un1]
  computed_stages = [ cg.vector_stage(sname) for sname in "1,2,".split(",") ]
  dt_stages = [ 0. , 1.0 , 0.5 ]

  return lawson( computed_stages , expr_stages , dt_stages , is_embeded=False )


methods = { "RK44":LRK44, "DP43":LDP43 , "RK33":LRK33, "Euler":LRK11 }

