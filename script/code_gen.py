#! /usr/bin/env python3

"""{exe}

Usage:
  ./{exe} (--pade N [M] | --taylor N | --exp ) [--maxwell] [--output-dir=<folder>]

Options:
  -h, --help                         Help! I need somebody
  -p N [M], --pade N [M]             Write code of Lawson method with Pade approximant of order [N,M] (by default M=N)
  -t N, --taylor N                   Write code of Lawson method with Taylor serie to order N
  -e, --exp                          Write code of Lawson method with a classical exponential (this is incompatible with --maxwell option)
  -m, --maxwell                      If define, the matrix of the linear part contain Maxwell equations
  -o=<folder>,--output-dir=<folder>  Define output directory for output simulation source code (current directory by default)
"""

import dataclasses

import sympy as sp
import numpy as np

# TODO (pas urgent du tout): trouver un moyen de passer ça argument si on veut d'autres valeurs
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

### approximant exponential
def taylor(n,):
  r"""
    return Taylor series of order $n$
  """
  def r_lambda(M):
    _N = M[:-1,:-1]
    nb_rows,nb_cols = _N.shape
    fac = sp.factorial
    eN = sum([
        _N**i/fac(i)
        for i in range(0,n+1)
      ],start=sp.zeros(*_N.shape))
    return sp.Matrix(sp.BlockMatrix([
        [ eN , sp.zeros(nb_rows,1) ],
        [ sp.zeros(1,nb_cols) , sp.Matrix([sp.exp(M[-1,-1])]) ]
      ]))

  return r_lambda

def submat(M,i,j):
  """
    return a copy of matrix `M` whitout row `i` and column `j`
  """
  N = M.copy()
  N.row_del(i)
  N.col_del(j)
  return N
def det(M,i=0,j=0):
  """
    compute determinant of M with naive implementation
  """
  n,m = M.shape
  if n==2 and m==2 :
    d=M[0,0]*M[1,1] - M[1,0]*M[0,1]
    return d
  return sum([
      ((-1)**( i+i_row+j ))*M[i_row,0]*det(submat(M,i_row,0))
      for i_row in range(n)
    ])

def invert(M):
  r"""
    invert a matrix `M` with naive formula :
    $$
      M^{-1} = \frac{1}{\det(M)}^t\textrm{com}(M)
    $$
  """
  import itertools
  n,m = M.shape
  invM = sp.zeros(*M.shape)
  for i,j in itertools.product(range(n),range(m)):
    tmp = M.copy()
    tmp.row_del(i)
    tmp.col_del(j)
    invM[j,i] = (-1)**(i+j)*det(tmp)
  return invM/(det(M))

def pade(n,m):
  r"""
    return Padé approximant of order $(n,m)$
  """
  def _h(p,q):
    r"""
      return numerator of rational approximant of Padé
    """
    fac = sp.factorial
    return lambda M : sum([
      ((fac(p)/fac(p-i)).simplify()/(fac(p+q)/fac(p+q-i)).simplify()*M**i/fac(i))
      for i in range(0,p+1)
    ],start=sp.zeros(*M.shape))

  def _k(p,q):
    r"""
      return denominator of rational approximant of Padé
    """
    fac = sp.factorial
    return lambda M : sum([
      (-1)**j*(fac(q)/fac(q-j)).simplify()/(fac(p+q)/fac(p+q-j)).simplify()*M**j/fac(j)
      for j in range(0,q+1)
    ],start=sp.zeros(*M.shape))

  def r_lambda(M):
    _N = M[:-1,:-1]
    nb_rows,nb_cols = _N.shape
    hnm = _h(n,m)(_N).evalf()
    knm = _k(n,m)(_N).evalf()
    iknm = sp.refine(invert(knm),sp.Q.integer(k))
    eN = sp.refine(sp.simplify(hnm)*iknm,sp.Q.integer(k))
    return sp.Matrix(sp.BlockMatrix([
        [ eN                  , sp.zeros(nb_rows,1)           ],
        [ sp.zeros(1,nb_cols) , sp.Matrix([sp.exp(M[-1,-1])]) ]
      ]))

  return r_lambda

### functions to write vectorial scheme
def my_exp(L,t,k):
  """
    In fact we don't need to a `sp.MatrixSymbol` exponential
    We only need to return a matrix of function : (`matrix_ij(dt,k)`)_{i,j}
  """
  nb_cols,nb_rows = L.shape
  matExp = sp.Matrix(sp.BlockMatrix([
      [ sp.Matrix([ [ sp.Function("matrixExpr{}{}".format(i,j))(t,k) for j in range(nb_cols-1) ] for i in range(nb_rows-1) ]) , sp.zeros(nb_rows-1,1) ],
      [ sp.zeros(1,nb_cols-1) , sp.Matrix([ sp.exp(t*L[-1,-1]) ]) ]
    ]))
  return matExp

### to help to define stages
def space_variables_idx (stage_name):
  return [
    sp.symbols( "{var}{stage}[i]".format(var=var,stage=stage_name) )
    for var in ("hjcx","hjcy","hBx","hBy","hEx","hEy")
  ]
def phase_variables_idx (stage_name):
  return [
    sp.symbols( "{var}{stage}[k_x][k_y][k_z][i]".format(var=var,stage=stage_name) )
    for var in ["hf"]
  ]
def vectors_stages_idx ( stages_names ):
  return [
    sp.Matrix(
      space_variables_idx(stage) + phase_variables_idx(stage)
    )
    for stage in stages_names
  ]

def vectors_stages ( stages_names ):
  return [
    sp.Matrix([
      sp.symbols( "{var}{stage}".format(var=var,stage=stage) )
      for var in ("hjcx","hjcy","hBx","hBy","hEx","hEy","hf")
    ])
    for stage in stages_names
  ]

### function to convert scheme into code
@dataclasses.dataclass
class U_code:
  jcx:str = "hjcx"
  jcy:str = "hjcy"
  Bx:str = "hBx"
  By:str = "hBy"
  Ex:str = "hEx"
  Ey:str = "hEy"
  fh:str = "hfh"
  dt:float = 0.

  def keys():
    """
      returns name of attributs of the class not the contain of each string
    """
    return ('jcx','jcy','Bx','By','Ex','Ey','fh')

  def __getitem__(self, key):
    if isinstance(key, slice):
      indices = range(*key.indices(len(U_code.keys())))
      _keys = U_code.keys()
      return [ getattr(self,_keys[i]) for i in indices ]
    return getattr(self,U_code.keys()[key])

  def __setitem__(self, key, values):
    if isinstance(key, slice):
      indices = range(*key.indices(len(U_code.keys())))
      _keys = U_code.keys()
      for i in indices:
        setattr(self,_keys[i] , values[i])
    else:
      setattr(self,U_code.keys()[key] , values)

def expr_to_code (expr,symbols_replace,function_replace,display=None):
  import re

  if display is not None:
    print("+ expression to code {}".format(display),end="\r")

  # sympy function to replace into STL C++ functions
  math_to_stl = [(f,sp.Function("std::"+str(f),nargs=1)) for f in (sp.sin,sp.cos,sp.exp)]
  math_to_stl.append( (sp.sqrt,sp.Function("std::sqrt",nargs=1)) )
  
  # first step: replace symbols
  tmp = expr.subs(symbols_replace)

  # second step: use STL functions
  for old,new in math_to_stl:
    tmp = tmp.subs(old,new)
  
  # next step: use user-defined functions
  for old,new in function_replace:
    tmp = tmp.replace(old,new)

  # last step: convert all division by 2 by multiplication by 0.5
  a = sp.Wild('a')  
  tmp = tmp.replace(a/2,0.5*a,exact=True).evalf()
  
  # and return a string (where remove every `1.0*` pattern where there is not a number before)
  return re.sub(r"([^0-9])1\.0\*",r"\1",str(tmp))

def reduce_code(line):
  """
    a last patch to solve some issues
    replace every trivial power
    and remove last Python sqaure (`expr**2`)
  """
  import re

  # first remove all trivial power
  line = (line
    .replace("std::pow(1.0*t, 2)","t*t")
    .replace("std::pow(1.0*t, 3)","t*t*t")
    .replace("std::pow(1.0*t, 4)","*".join(["t"]*4))
    .replace("std::pow(1.0*t, 5)","*".join(["t"]*5))
    .replace("std::pow(1.0*t, 6)","*".join(["t"]*6))
    .replace("std::pow(1.0*k, 2)","k*k")
    .replace("std::pow(1.0*k, 3)","k*k*k")
    .replace("std::pow(1.0*k, 4)","*".join(["k"]*4))
    .replace("std::pow(1.0*k, 5)","*".join(["k"]*5))
    .replace("std::pow(1.0*k, 6)","*".join(["k"]*6))
    )

  # search all expression between brackets (without close bracket) and finish with `**2`
  # in short, search all Python square expression
  p = re.compile(r"\(([^)(]+)\)\*\*2")
  d = { m.group(0):"({expr}*{expr})".format(expr=m.group(0)[:-3]) for m in p.finditer(line) }
  for old,new in d.items() :
    line = line.replace(old,new)

  return line

def code_gen ( filename , simu_name , schemes , frange , expLt_mat ) :
  import jinja2

  env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
  template = env.get_template("hybrid_stalfos.jinja.cc")

  with open(filename,'w') as of:
    of.write(template.render(simu_name=simu_name,schemes=schemes,frange=frange,expLt_mat=expLt_mat))

def code_pow(x,e):
  """
    replace some trivial cases by a simpler expression (for code)
    because all division in sympy are `sp.Pow(expr,-1)` and same
    for every square, of square root
  """
  if e == 1 :
    return x
  if sp.S(e).is_number and e < 0 :
    return 1./(code_pow(x,-e))
  if e == sp.S.Half :
    return sp.Function("std::sqrt",nargs=1)(x)
  return sp.Function("std::pow",nargs=2)(x,e)

fun_to_code = [
  (Ix,lambda _:sp.symbols("hjhx[i]")),
  (Iy,lambda _:sp.symbols("hjhy[i]")),
  (df,lambda _:sp.symbols("hfvxvyvz[i]")),
  (sp.Pow,code_pow),
]

### main ############################################################

# import lib to manage arguments
import sys, os, docopt

if __name__ == '__main__':
  # use docopt to parse argv
  arg = docopt.docopt( __doc__.format(exe=sys.argv[0][2:]) )
  # select exponential computationner
  if arg['--exp'] :
    exp_meth_name = "".format(**arg)
    exp_meth = lambda M:sp.exp(M).simplify()
  elif arg['--taylor']:
    exp_meth_name = "t{--taylor}".format(**arg)
    exp_meth = taylor(int(arg['--taylor']))
  elif arg['--pade']:
    if arg['M'] is None:
      arg['M'] = arg['--pade']
    exp_meth_name = "p{--pade}{M}".format(**arg)
    exp_meth = pade(int(arg['--pade']),int(arg['M']))

  if arg['--output-dir'] is None:
    arg['--output-dir'] = "."

  # build simu_name (just missing RK parameters)
  simu_name_pattern = "vmhllf_{m}{exp_meth}rk{{s}}{{n}}".format(m="m" if arg['--maxwell'] else "",exp_meth=exp_meth_name)

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
  # vector of variables for each stage in scheme
  Un_idx, U1_idx, U2_idx, U3_idx = vectors_stages_idx( ["","1","2","3"] )
  Un    , U1    , U2    , U3     = vectors_stages( ["","1","2","3"] )

  eLt = my_exp(L,t,k)
  ### LRK(4,4) ######################################################
  order_rk = 4
  print("+ stage 1")
  stage_U1  =  eLt.subs(t,dt/2)*Un_idx + dt/2*eLt.subs(t,dt/2)*N(Un_idx)
  print("+ stage 2")
  stage_U2  =  eLt.subs(t,dt/2)*Un_idx + dt/2*N(U1_idx)
  print("+ stage 3")
  stage_U3  =  eLt.subs(t,dt)*Un_idx   + dt*eLt.subs(t,dt/2)*N(U2_idx)
  print("+ stage n+1")
  stage_Un1 = -eLt.subs(t,dt)*Un_idx/3 + eLt.subs(t,dt/2)*U1_idx/3 + 2*eLt.subs(t,dt/2)*U2_idx/3 + U3_idx/3 + dt/6*N(U3_idx)

  stages = [stage_U1,stage_U2,stage_U3,stage_Un1]
  computed_stage = [U1,U2,U3,Un]
  stages_dt = [ 0. , 0.5 , 0.5 , 1.0 ]
  ###################################################################

  print("> compute exponential approximant")
  expLt_mat = exp_meth(t*L)

  print("> code generation")
  # appeler une fonction qui prend les étage et renvoie `list_stages`
  list_stages = []
  for i,(stage,nextU,stage_dt) in enumerate(zip(stages,computed_stage,stages_dt)):
    print("+ stage ",i)
    lhs,rhs = (U_code(),U_code())

    lhs[:] = [ str(ui) for ui in nextU ]
    lhs.dt = stage_dt

    rhs[:] = [ expr_to_code(line,{k:sp.symbols("Kz[i]")},fun_to_code) for line in stage ]

    list_stages.append( (lhs,rhs) )

  print("> matrix code generation")
  expLt_cmat = [
    [
      reduce_code( expr_to_code(expLt_mat[i,j],{},fun_to_code,"matrix [{},{}]".format(i,j)) )
      for j in range(expLt_mat.shape[1]-1)
    ]
    for i in range(expLt_mat.shape[0]-1)
  ]
  print()

  print("> code printing")
  simu_name = simu_name_pattern.format(s=len(list_stages),n=order_rk)
  filename = os.path.join(arg['--output-dir'],"hybrid_{simu_name}.cc".format(simu_name=simu_name))
  print(simu_name)
  code_gen( filename , simu_name , list_stages , f_range() , expLt_cmat )
  print("Finish!")
