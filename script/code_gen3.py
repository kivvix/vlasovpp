#! /usr/bin/env python3

"""{exe}

Usage:
  ./{exe} (--pade N [M] | --taylor N | --exp ) [--maxwell] [--lrk=<method>] [--output-dir=<folder>]

Options:
  -h, --help                         Help! I need somebody
  -p N [M], --pade N [M]             Write code of Lawson method with Pade approximant of order [N,M] (by default M=N)
  -t N, --taylor N                   Write code of Lawson method with Taylor serie to order N
  -e, --exp                          Write code of Lawson method with a classical exponential (this is incompatible with --maxwell option)
  -m, --maxwell                      If define, the matrix of the linear part contain Maxwell equations
  --lrk=<method>                     Select a Lawson-Runge-Kutta method in {meths}, by default it is RK44
  -o=<folder>,--output-dir=<folder>  Define output directory for output simulation source code (current directory by default)
"""

import sympy as sp
import dataclasses

import cgeneration as cg
import schemes
import mexp

# TODO (pas urgent du tout): trouver un moyen de passer ça par argument si on veut d'autres valeurs
@dataclasses.dataclass
class f_range:
  vx_max =  3.6
  vx_min = -3.6
  vy_max =  3.6
  vy_min = -3.6
  vz_max =  2.0
  vz_min = -2.0

######################################################################

k  = sp.symbols("k",real=True)
t  = sp.symbols("t",real=True)
dt = sp.symbols("dt",real=True)
wpe = 2
vx,vy,vz = sp.symbols("v_x v_y v_z",real=True)

Ix = sp.Function(r"\int_\mathbb{R}\ v_x")
Iy = sp.Function(r"\int_\mathbb{R}\ v_y")
df = sp.Function(r"(E+v\times B)\cdot\nabla_v")

fun_to_code = [
  (Ix,lambda _:sp.symbols("hjhx[i]")),
  (Iy,lambda _:sp.symbols("hjhy[i]")),
  (df,lambda _:sp.symbols("hfvxvyvz[i]")),
  (sp.Pow,cg.code_pow),
]

### main #############################################################

# import lib to manage arguments
import sys, os, docopt

if __name__ == '__main__':
  # use docopt to parse argv
  arg = docopt.docopt( __doc__.format(exe=sys.argv[0][2:],meths=", ".join([key for key in schemes.methods])) )
  # select exponential computationner
  if arg['--exp'] :
    exp_meth_name = "".format(**arg)
    exp_meth = lambda M:sp.exp(M).simplify()
  elif arg['--taylor']:
    exp_meth_name = "t{--taylor}".format(**arg)
    exp_meth = mexp.taylor(int(arg['--taylor']))
  elif arg['--pade']:
    if arg['M'] is None:
      arg['M'] = arg['--pade']
    exp_meth_name = "p{--pade}{M}".format(**arg)
    exp_meth = mexp.pade(int(arg['--pade']),int(arg['M']))

  if arg['--output-dir'] is None:
    arg['--output-dir'] = "."

  # build simu_name (just missing RK label)
  simu_name_pattern = "vmhllf_{m}{exp_meth}{{lab}}".format(m="m" if arg['--maxwell'] else "",exp_meth=exp_meth_name)

  # select linear part L and nonlinear part N from --maxwell argument
  if arg['--maxwell']:
    L = sp.Matrix([
        [ 0 , -1 , 0      ,  0      ,  wpe**2 , 0      ,  0         ],
        [ 1 ,  0 , 0      ,  0      ,  0      , wpe**2 ,  0         ],
        [ 0 ,  0 , 0      ,  0      ,  0      , sp.I*k ,  0         ],
        [ 0 ,  0 , 0      ,  0      , -sp.I*k , 0      ,  0         ],
        [-1 ,  0 , 0      , -sp.I*k ,  0      , 0      ,  0         ],
        [ 0 , -1 , sp.I*k ,  0      ,  0      , 0      ,  0         ],
        [ 0 ,  0 , 0      ,  0      ,  0      , 0      , -sp.I*k*vz ]
      ])
    def N(U):
      jcx,jcy,Bx,By,Ex,Ey,f = (*U,)
      return sp.Matrix([
        0,
        0,
        0,
        0,
        Ix(f),
        Iy(f),
        df(f)
      ])
  else:
    L = sp.Matrix([
        [ 0 , -1 , 0 , 0 , wpe**2 , 0      ,  0         ],
        [ 1 ,  0 , 0 , 0 , 0      , wpe**2 ,  0         ],
        [ 0 ,  0 , 0 , 0 , 0      , 0      ,  0         ],
        [ 0 ,  0 , 0 , 0 , 0      , 0      ,  0         ],
        [-1 ,  0 , 0 , 0 , 0      , 0      ,  0         ],
        [ 0 , -1 , 0 , 0 , 0      , 0      ,  0         ],
        [ 0 ,  0 , 0 , 0 , 0      , 0      , -sp.I*k*vz ]
      ])
    def N(U):
      jcx,jcy,Bx,By,Ex,Ey,f = (*U,)
      return sp.Matrix([
        0,
        0,
        sp.I*k*Ey,
        -sp.I*k*Ex,
        -sp.I*k*By+Ix(f),
        sp.I*k*Bx+Iy(f),
        df(f)
      ])

  print("> scheme generation")
  eLt = cg.funcMatExp(L,t,k)
  # TODO (un peu plus urgent): trouver un moyen de passer ça (ou le tableau de Butcher) par argument (fichier ?) ou relier ça à Ponio
  if arg["--lrk"] is None :
    arg["--lrk"] = "RK44"
  if arg["--lrk"] not in schemes.methods :
    print("Lawson method not in:\n\t{}".format(" ".join(k for k in schemes.methods)))
    sys.exit()
  print(arg["--lrk"])

  lrk = schemes.methods[arg["--lrk"]]( t,dt,eLt,N  )

  ####################################################################

  print("> compute exponential approximant")
  expLt_mat = exp_meth(t*L)

  print("> code generation")
  for (lhs,rhs) in lrk.stages :
    lhs[:] = [ str(ui) for ui in lhs[:] ]
    rhs[:] = [ cg.expr_to_code(ui_expr,{k:sp.symbols("Kz[i]")},fun_to_code) for ui_expr in rhs[:] ]

  print("> matrix code generation")
  expLt_cmat = [
    [
      cg.reduce_code( cg.expr_to_code(expLt_mat[i,j],{},fun_to_code,"matrix [{},{}]".format(i,j)) )
      for j in range(expLt_mat.shape[1]-1)
    ]
    for i in range(expLt_mat.shape[0]-1)
  ]
  print()

  print("> code printing")
  simu_name = simu_name_pattern.format(lab=arg["--lrk"].lower())
  filename = os.path.join(arg['--output-dir'],"hybrid_{simu_name}.cc".format(simu_name=simu_name))
  print(simu_name)
  cg.code_gen( filename , simu_name , lrk.stages , f_range() , expLt_cmat , lrk.is_embeded , lrk.un )
  print("Finish!")
