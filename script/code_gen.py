#! /usr/bin/env python

import os, sys
import jinja2
import dataclasses

import sympy as sp
import numpy as np

k = sp.symbols("k")
wpe = 2
vx,vy,vz = sp.symbols("v_x v_y v_z",real=True)

L = sp.Matrix([
    [ 0 , -1 , 0 , 0 , wpe**2 , 0      ,  0         ],
    [ 1 ,  0 , 0 , 0 , 0      , wpe**2 ,  0         ],
    [ 0 ,  0 , 0 , 0 , 0      , 0      ,  0         ],
    [ 0 ,  0 , 0 , 0 , 0      , 0      ,  0         ],
    [-1 ,  0 , 0 , 0 , 0      , 0      ,  0         ],
    [ 0 , -1 , 0 , 0 , 0      , 0      ,  0         ],
    [ 0 ,  0 , 0 , 0 , 0      , 0      , -sp.I*k*vz ]
  ])

Ix = sp.Function(r"\int_\mathbb{R}\ v_x")
Iy = sp.Function(r"\int_\mathbb{R}\ v_y")
df = sp.Function(r"(E+v\times B)\cdot\nabla_v")

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

# order of Pade approximant
order_pade = 2

######################################################################
######################################################################
######################################################################

r"""
In first we define pade approximant with some sugar syntax to help to
 write LRK scheme.

The main goal of opptimize generation of numerical scheme is to eval
 once Pade approximant of linear part and convert it into a function
 in $t$ :
 $$
   P_{[p.q]}^{A}:t\mapsto P_{[p,q]}(tA) \approx e^{tA}
 $$
 to make it we compute Padé approximant of $e^{tA}$ to user defined
 order, and next substitue $t$ by $\alpha_{j}\Delta t$ coefficients
 from Butcher tableau.

After generation of scheme with `sympy`, we need to convert it into
 C++ code (this is make with substitution in `sympy` expressions) and
 take it in a skeleton of C++ simulation code (with a template engine,
 I use `jinja2` for this).
"""

def is_matrix(z):
  r"""
    return True if `z` is a `sympy`'s matrix (or something like a
     matrix like `sp.MatrixSymbol` or `sp..MatrixExpr`)
  """
  return ( isinstance(z,sp.Matrix)
        or isinstance(z,sp.ImmutableDenseMatrix)
        or isinstance(z,sp.MatrixSymbol)
        or isinstance(z,sp.MatrixExpr)
         )
def one(z):
  r"""
    return unit element (1 if `z` is a scalar or `sp.eye` if `z` is a
     matrix)
  """
  if is_matrix(z):
    return sp.eye(z.cols)
  return 1
def zero(z):
  r"""
    return zero element (0 if `z` is a scalar or `sp.zeros` if `z` is
     a matrix)
  """
  if is_matrix(z):
    return sp.zeros(z.cols)
  return 0
def inv(z):
  r"""
    return inverse element ($\frac{1}{z}$ if `z` is a scalar, $z^{-1}$
     if `z` is a matrix)
  """
  if is_matrix(z):
    return z.inv()
  return 1/z

def pade(n,m):
  r"""
    return Padé approximant of order $(n,m)$
  """
  def h(p,q):
    r"""
      return numerator of rational approximant of Padé
    """
    fac = sp.factorial
    return lambda x : sum([
      ((fac(p)/fac(p-i)).simplify()/(fac(p+q)/fac(p+q-i)).simplify()*one(x)*x**i/fac(i))
      for i in range(0,p+1)
    ],start=zero(x))

  def k(p,q):
    r"""
      return denominator of rational approximant of Padé
    """
    fac = sp.factorial
    return lambda x : sum([
      (-1)**j*(fac(q)/fac(q-j)).simplify()/(fac(p+q)/fac(p+q-j)).simplify()*one(x)*x**j/fac(j)
      for j in range(0,q+1)
    ],start=zero(x))
  
  return lambda z: h(n,m)(z)*inv(k(n,m)(z))


def ms_exp(expr):
  """
    matrix symbol exponential

    I wrote this function because `sympy` doesn't work with
     exponential of a `sp.MatrixSymbol` (it returns a scalar...). This
     is not really easy to use, but it works with some sugar functions
     it does the job. Next I could replace every `sp.MatrixSymbol`
     with any exponential function (real exponential function of Padé
     approximant).
  """
  mexp = sp.MatrixSymbol((r"e^{"+sp.latex(expr)+r"}").replace(" ",r"\ "),*(expr).shape)
  mexp.arg = expr
  return mexp

def ems_args(*arg):
  """
    exponential matrix symbol arguments
  """
  s = str(arg[0])[3:-1].replace(r"\ "," ").replace(r"\left(","").replace(r"\right)","")
  from sympy.parsing.latex import parse_latex
  return parse_latex(s)

def ems_subs(A,B):
  """
    exponential matrix symbol substitution
    substitue matrix A into B in arguments of an ems
    return lambda which only returns arguments substituion
  """
  return lambda *arg : ems_args(*arg).subs(sp.symbols(str(A)),B)

def ems_func(lamb,func):
  """
    exponential matrix symbol function
    get an already define lambda (lamb) substitution
    and apply user define function (func)
  """
  return lambda *arg : func(lamb(*arg))


######################################################################
# We write our scheme here ###########################################
######################################################################

r"""
compute scheme with a symbolic matrix `sm_L`
"""
dt = sp.symbols("dt")
sm_L = sp.MatrixSymbol("L",*L.shape)

jxn,jyn = sp.symbols(r"j_{c\,x}^n j_{c\,y}^n")
Bxn,Byn = sp.symbols(r"B_x^n B_y^n")
Exn,Eyn = sp.symbols(r"E_x^n E_y^n")
fn = sp.symbols(r"\hat{f}^n")
Un = sp.Matrix([jxn,jyn,Bxn,Byn,Exn,Eyn,fn])

jx1,jy1 = sp.symbols(r"j_{c\,x}^{(1)} j_{c\,y}^{(1)}")
Bx1,By1 = sp.symbols(r"B_x^{(1)} B_y^{(1)}")
Ex1,Ey1 = sp.symbols(r"E_x^{(1)} E_y^{(1)}")
f1 = sp.symbols(r"\hat{f}^{(1)}")
U1 = sp.Matrix([jx1,jy1,Bx1,By1,Ex1,Ey1,f1])

jx2,jy2 = sp.symbols(r"j_{c\,x}^{(2)} j_{c\,y}^{(2)}")
Bx2,By2 = sp.symbols(r"B_x^{(2)} B_y^{(2)}")
Ex2,Ey2 = sp.symbols(r"E_x^{(2)} E_y^{(2)}")
f2 = sp.symbols(r"\hat{f}^{(2)}")
U2 = sp.Matrix([jx2,jy2,Bx2,By2,Ex2,Ey2,f2])

jx3,jy3 = sp.symbols(r"j_{c\,x}^{(3)} j_{c\,y}^{(3)}")
Bx3,By3 = sp.symbols(r"B_x^{(3)} B_y^{(3)}")
Ex3,Ey3 = sp.symbols(r"E_x^{(3)} E_y^{(3)}")
f3 = sp.symbols(r"\hat{f}^{(3)}")
U3 = sp.Matrix([jx3,jy3,Bx3,By3,Ex3,Ey3,f3])

print("> scheme generation")

# LRK(4,4)
print("+ stage 1")
stage_U1  =  ms_exp(dt/2*sm_L)*Un + dt/2*ms_exp(dt/2*sm_L)*N(Un)
print("+ stage 2")
stage_U2  =  ms_exp(dt/2*sm_L)*Un + dt/2*N(U1)
print("+ stage 3")
stage_U3  =  ms_exp(dt*sm_L)*Un   + dt*ms_exp(dt/2*sm_L)*N(U2)
print("+ stage n+1")
stage_Un1 = -ms_exp(dt*sm_L)*Un/3 + ms_exp(dt/2*sm_L)*U1/3 + 2*ms_exp(dt/2*sm_L)*U2/3 + U3/3 + dt/6*N(U3)

dts = [ 0. , 0.5 , 0.5 , 1.0 ]

r"""
we compute Pade approximant of $tL$
"""

print("> compute Pade approximant ({})".format(order_pade))
t = sp.symbols("t",real=True,positive=True)
Pt = pade(order_pade,order_pade)(t*L)

def Pt_func(X):
  dt = X[1,0]/L[1,0] # normalement le terme L[1,0] est non nul, il faut juste un terme non nul pour retrouver dt
  return Pt.subs(t,dt)

r"""
replace $e{\alpha_i\Delta t L}$ by Padé approximant with smart
substituion
"""

print("> substitute in scheme")

print("+ stage 1")
pade_stage_U1 = stage_U1.replace(
    sp.MatrixSymbol,
    ems_func(ems_subs(sm_L,L),Pt_func)
  ).replace(sp.MatMul,sp.Mul).replace(sp.MatAdd,sp.Add)

print("+ stage 2")
pade_stage_U2 = stage_U2.replace(
    sp.MatrixSymbol,
    ems_func(ems_subs(sm_L,L),Pt_func)
  ).replace(sp.MatMul,sp.Mul).replace(sp.MatAdd,sp.Add)

print("+ stage 3")
pade_stage_U3 = stage_U3.replace(
    sp.MatrixSymbol,
    ems_func(ems_subs(sm_L,L),Pt_func)
  ).replace(sp.MatMul,sp.Mul).replace(sp.MatAdd,sp.Add)

print("+ stage n+1")
pade_stage_Un1 = stage_Un1.replace(
    sp.MatrixSymbol,
    ems_func(ems_subs(sm_L,L),Pt_func)
  ).replace(sp.MatMul,sp.Mul).replace(sp.MatAdd,sp.Add)

######################################################################
# We convert scheme into code ########################################
######################################################################

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

@dataclasses.dataclass
class f_range:
  vx_max =  3.6
  vx_min = -3.6
  vy_max =  3.6
  vy_min = -3.6
  vz_max =  2.0
  vz_min = -2.0

def expr_to_code (expr,symbols_replace,function_replace):
  # sympy function to replace into STL C++ functions
  math_to_stl = [(f,sp.Function("std::"+str(f),nargs=1)) for f in (sp.sin,sp.cos,sp.exp)]
  math_to_stl.append( (sp.sqrt,sp.Function("std::sqrt",nargs=1)) )

  a = sp.Wild('a')
  def func(expr):
    print(expr is sp.S.Half,expr)
    return sp.Float(0.5)
  
  # first step: replace symbols
  #try:
  #  tmp = sp.simplify(expr.subs(symbols_replace))
  #except:
  tmp = expr.subs(symbols_replace)
  
  # second step: use STL functions
  for old,new in math_to_stl:
    tmp = tmp.subs(old,new)
  
  # next step: use user-defined functions
  for old,new in function_replace:
    tmp = tmp.replace(old,new)

  # last step: convert all division by 2 by multiplication by 0.5
  tmp = tmp.replace(a/2,0.5*a,exact=True).evalf()
  
  # and return a string
  return str(tmp)

dt_sym_to_str = [
  (dt,"dt")
]
space_sym_to_str = [
  (k,"{}[i]".format(v))
  for (k,v) in [
    (k,"Kz") ,
    (jxn, "hjcx") , (jyn, "hjcy") ,
    (Bxn,  "hBx") , (Byn,  "hBy") ,
    (Exn,  "hEx") , (Eyn,  "hEy") ,
    (jx1,"hjcx1") , (jy1,"hjcy1") ,
    (Bx1, "hBx1") , (By1, "hBy1") ,
    (Ex1, "hEx1") , (Ey1, "hEy1") ,
    (jx2,"hjcx2") , (jy2,"hjcy2") ,
    (Bx2, "hBx2") , (By2, "hBy2") ,
    (Ex2, "hEx2") , (Ey2, "hEy2") ,
    (jx3,"hjcx3") , (jy3,"hjcy3") ,
    (Bx3, "hBx3") , (By3, "hBy3") ,
    (Ex3, "hEx3") , (Ey3, "hEy3") ,
  ]
]
phase_sym_to_str = [
  (k,"{}[k_x][k_y][k_z][i]".format(v))
  for (k,v) in [
    ( fn,   "hf") ,
    ( f1,  "hf1") ,
    ( f2,  "hf2") ,
    ( f3,  "hf3") ,
  ]
]

sym_to_code = [
  (k,sp.symbols(v))
  for (k,v) in [*dt_sym_to_str,*space_sym_to_str,*phase_sym_to_str]
]

def code_pow(x,e):
  """
    replace some trivial cases by a simpler expression (for code)
  """
  if e == 1 :
    return x
  if e < 0 :
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

print("> code generation")
list_stages = []
for stage,nextU,adt in zip([pade_stage_U1,pade_stage_U2,pade_stage_U3,pade_stage_Un1],[U1,U2,U3,Un],dts) :
  code_nextU = nextU.subs(sym_to_code)

  lhs,rhs = (U_code(),U_code())

  lhs[:-1] = [ str(ui)[:-3] for ui in code_nextU[:-1] ]
  lhs[-1]  = str(code_nextU[-1])[:-18]
  lhs.dt   = adt
  rhs[:] = [ expr_to_code(line,sym_to_code,fun_to_code) for line in stage ]
  #rhs[:] = [ "0.+0." for _ in stage ]

  list_stages.append( (lhs,rhs) )

print("> code printing")
frange = f_range()
env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
template = env.get_template("hybrid_stalfos.t.cc")

simu_name = "vmhllf_p{}rk{}{}".format(order_pade,len(list_stages),len(list_stages))

with open("hybrid_{}.cc".format(simu_name),'w') as of:
  of.write(template.render(simu_name=simu_name,schemes=list_stages,frange=frange))



