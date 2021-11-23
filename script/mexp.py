#! /usr/bin/env python3

import sympy as sp

r"""
# Approximant exponential

We propose two approximation of exponential of a matrix. This two
approximations are available only for a matrix of the shape :
$$
  \begin{pmatrix}
    A & 0 \\
    0 & x
  \end{pmatrix}
$$
with $A$ a submatrix, and $x$ a scalar. That why `taylor` and `pade`
functions begin by `_N = M[:-1,:-1]` to extract the submatrix $A$.
"""

# Exact exponential
def expm(A):
  r"""
    here we use the special form of matrix M to compute its exponential
  """
  eA = sp.exp(A[:-1,:-1]).evalf().simplify().row_join(sp.zeros(6,1)).col_join(sp.Matrix(1,7,[0]*6+[sp.exp(A[-1,-1])]))
  return eA.simplify()

# Polynomial or rational approximations

## Taylor series
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

## Pade approximants
def _submat(M,i,j):
  """
    return a copy of matrix `M` whitout row `i` and column `j`
  """
  N = M.copy()
  N.row_del(i)
  N.col_del(j)
  return N
def _det(M,i=0,j=0):
  """
    compute determinant of M with naive implementation
  """
  n,m = M.shape
  if n==2 and m==2 :
    d=M[0,0]*M[1,1] - M[1,0]*M[0,1]
    return d
  return sum([
      ((-1)**( i+i_row+j ))*M[i_row,0]*_det(_submat(M,i_row,0))
      for i_row in range(n)
    ])

def _invert(M):
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
    invM[j,i] = (-1)**(i+j)*_det(tmp)
  return invM/(_det(M))

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
    k  = sp.symbols("k",real=True)
    iknm = sp.refine(_invert(knm),sp.Q.integer(k))
    eN = sp.refine(sp.simplify(hnm)*iknm,sp.Q.integer(k))
    return sp.Matrix(sp.BlockMatrix([
        [ eN                  , sp.zeros(nb_rows,1)           ],
        [ sp.zeros(1,nb_cols) , sp.Matrix([sp.exp(M[-1,-1])]) ]
      ]))

  return r_lambda

# Splitting approximation
def strang(M):
  r"""
    A Strang method to approximate exponential:
    $$
      exp(t(A_1+A_2)) \approx exp(t/2 A_1)exp(tA_2)exp(t/2 A_1)
    $$
  """
  k  = sp.symbols("k",real=True)

  A = M[:-1,:-1]
  nb_rows,nb_cols = A.shape
  A1 = A.subs(k,0)
  A2 = A-A1

  tau = A[1,0] # because B_0 = 1
  eA1 = sp.exp(A1).evalf().simplify()
  eA2 = sp.exp(A2)

  Strang = eA2.subs(tau,0.5*tau)*eA1*eA2.subs(tau,0.5*tau)

  return sp.Matrix(sp.BlockMatrix([
        [ Strang              , sp.zeros(nb_rows,1)           ],
        [ sp.zeros(1,nb_cols) , sp.Matrix([sp.exp(M[-1,-1])]) ]
      ]))

def suzuki(M):
  r"""
    A Suzuki method to approximate exponential:
    $$
      exp(t(A_1+A_2)) \approx \dots
    $$
  """
  k  = sp.symbols("k",real=True)

  A = M[:-1,:-1]
  nb_rows,nb_cols = A.shape
  A1 = A.subs(k,0)
  A2 = A-A1

  tau = A[1,0] # because B_0 = 1
  eA1 = sp.exp(A1).evalf().simplify()
  eA2 = sp.exp(A2)

  from numpy import cbrt
  a1 = 1.0/(4.0 - cbrt(4.0))
  a2 = a1
  a3 = 1.0/(4.0 - cbrt(4.0)**2)

  Strang = eA2.subs(tau,0.5*tau)*eA1*eA2.subs(tau,0.5*tau) # one Strang
  St1 = Strang.subs(tau,a1*tau).evalf().simplify()
  St2 = Strang.subs(tau,a2*tau).evalf().simplify()
  St3 = Strang.subs(tau,a3*tau).evalf().simplify()

  Suzuki = St1*St2*St3*St2*St1

  return sp.Matrix(sp.BlockMatrix([
        [ Suzuki.evalf()      , sp.zeros(nb_rows,1)           ],
        [ sp.zeros(1,nb_cols) , sp.Matrix([sp.exp(M[-1,-1])]) ]
      ]))


#TODO: BCH formula
