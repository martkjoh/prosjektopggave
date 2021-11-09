import sympy as sp

d, R, c = sp.symbols('d R c')

b = sp.sqrt(c**2+d)

print(sp.simplify(b))
