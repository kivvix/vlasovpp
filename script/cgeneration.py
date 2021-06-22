#! /usr/bin/env python3

import sympy as sp
import dataclasses

r"""
# Code generation

## Matrix manupulation

In the scheme we only manipulate a matrix of IT functions instead of
the exponential of the matrix: $e^{tL}$. We know the shape of $L$:
$$
  \begin{pmatrix}
    A & 0 \\
    0 & x
  \end{pmatrix}
$$
with $A$ a submatrix, and $x$ a scalar, so the exponential has the
same shape. The matrix of IT functions will be:
$$
  \begin{pmatrix}
    \texttt{B} & 0 \\
    0 & e^{tx}
  \end{pmatrix}
$$
with $B$ a matrix like:
$$
  B_{i,j} = \texttt{matrixExprij}( t , k )
$$
with `t` and `k` two arguments of our function (for time and Fourier
mode).
"""

def funcMatExp(L,t,k):
  """
    In fact we don't need to compute the real $e^{tL}$ with a classic
    exponential function, or with Pade approximant or Taylor serie, we
    only need to manipulate the matrix of IT functions.
    This function returns the matrix of IT functions.
  """
  nb_cols,nb_rows = L.shape
  B = sp.Matrix([
      [
        sp.Function( "matrixExpr{}{}".format(i,j) ) (t,k)
        for j in range(nb_cols-1)
      ]
      for i in range(nb_rows-1)
    ])
  matExp = sp.Matrix(sp.BlockMatrix([
      [ B                     , sp.zeros(nb_rows-1,1)             ],
      [ sp.zeros(1,nb_cols-1) , sp.Matrix([ sp.exp(t*L[-1,-1]) ]) ]
    ]))
  return matExp

r"""
## Vectors generation

For each stage we have a vector of 7 variables :
$$
  \begin{pmatrix}
    \hat{j}_{c,x} \\
    \hat{j}_{c,y} \\
    \hat{B}_x \\
    \hat{B}_y \\
    \hat{E}_x \\
    \hat{E}_y \\
    \hat{f}_h
  \end{pmatrix}
$$
6 first variables depend only of space (index `i`), and the last is a
phase space variable, depends of velocity (3d) and space (1d) (index
`k_x`, `k_y`, `k_z` and `i`).
"""

### to help to define stages
def space_variables_idx (stage_name):
  """
    returns a list of symbols for each space variable of one stage
  """
  return [
    sp.symbols( "{var}{stage}[i]".format(var=var,stage=stage_name) )
    for var in ("hjcx","hjcy","hBx","hBy","hEx","hEy")
  ]
def phase_variables_idx (stage_name):
  r"""
    returns a list of symbols for $\hat{f}$ of one stage
  """
  return [
    sp.symbols( "{var}{stage}[k_x][k_y][k_z][i]".format(var=var,stage=stage_name) )
    for var in ["hf"]
  ]

def vector_stage_idx ( stage_name ):
  """
    returns a vector of all 7 variables of one stage with all indexes
  """
  return sp.Matrix(
      space_variables_idx(stage_name) + phase_variables_idx(stage_name)
    )

def vector_stage ( stage_name ):
  """
    returns the same vector of all 7 variables of one stage than
    `vectors_stages_idx` without indexes
  """
  return sp.Matrix([
      sp.symbols( "{var}{stage}".format(var=var,stage=stage_name) )
      for var in ("hjcx","hjcy","hBx","hBy","hEx","hEy","hf")
    ])

r"""
## Code generation

For each stage, we define from the Lawson scheme a *lhs* value (value
with the stage name, this is a vector with `hjcx1` for exemple) and a
*rhs* value (the expression with matrix product and all this sutff).

For each stage the Jinja template allows us to write something like:

```
  lhs[i] = rhs[i]
```

where `lhs[0]` represent $\hat{j}_{c,x}$ of the considered stage, and
`rhs[0]` the correspondant expression to compute it (and same for
other value of `i`).

And next convert each `rhs[i]` into a valid C++ expression.
"""

@dataclasses.dataclass
class Uhs:
  """
    Undefined hand side (for lhs and rhs)
  """
  jcx = "hjcx"
  jcy = "hjcy"
  Bx  = "hBx"
  By  = "hBy"
  Ex  = "hEx"
  Ey  = "hEy"
  fh  = "hfh"
  dt:float = 0.

  def keys():
    """
      returns name of attributs of the class not the contain of each string
    """
    return ('jcx','jcy','Bx','By','Ex','Ey','fh')

  def __getitem__(self, key):
    """
      to get item from it's short name with bracket operator, and it
      allows to get a range of values
    """
    if isinstance(key, slice):
      indices = range(*key.indices(len(Uhs.keys())))
      _keys = Uhs.keys()
      return [ getattr(self,_keys[i]) for i in indices ]
    return getattr(self,Uhs.keys()[key])

  def __setitem__(self, key, values):
    """
      to set item from it's short name with bracket operator, and it
      allows to set a range of values
    """
    if isinstance(key, slice):
      indices = range(*key.indices(len(Uhs.keys())))
      _keys = Uhs.keys()
      for i in indices:
        setattr(self,_keys[i] , values[i])
    else:
      setattr(self,Uhs.keys()[key] , values)

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
    .replace("std::pow(t, 2)","t*t")
    .replace("std::pow(t, 3)","t*t*t")
    .replace("std::pow(t, 4)","*".join(["t"]*4))
    .replace("std::pow(t, 5)","*".join(["t"]*5))
    .replace("std::pow(t, 6)","*".join(["t"]*6))
    .replace("std::pow(k, 2)","k*k")
    .replace("std::pow(k, 3)","k*k*k")
    .replace("std::pow(k, 4)","*".join(["k"]*4))
    .replace("std::pow(k, 5)","*".join(["k"]*5))
    .replace("std::pow(k, 6)","*".join(["k"]*6))
    )

  # search all expression between brackets (without close bracket) and finish with `**2`
  # in short, search all Python square expression
  p = re.compile(r"\(([^)(]+)\)\*\*2")
  d = { m.group(0):"({expr}*{expr})".format(expr=m.group(0)[:-3]) for m in p.finditer(line) }
  for old,new in d.items() :
    line = line.replace(old,new)

  return line

def code_gen ( filename , simu_name , schemes , frange , expLt_mat , is_embeded=False , un=None) :
  import jinja2

  env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
  template = env.get_template("hybrid_jack.jinja.cc")

  with open(filename,'w') as of:
    of.write(template.render(simu_name=simu_name,schemes=schemes,frange=frange,expLt_mat=expLt_mat,is_embeded=is_embeded,un=un))

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


