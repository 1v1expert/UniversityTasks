from sympy import symbols, Rational, diff
from sympy.core.add import Add


class KernelFunction:
    def __init__(
        self,
        a: Rational = Rational(0),
        b: Rational = Rational(0),
        c: Rational = Rational(0),
        d: Rational = Rational(0),
        e: Rational = Rational(0),
    ):
        self.x, self.y = symbols('x y')
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

    @property
    def H(self) -> Add:
        return (self.a * self.x ** 2) + (self.b * self.y ** 2) + \
               (self.c * self.x * self.y) + \
               (self.d * self.x) + (self.e * self.y)

    @property
    def dH_dx(self) -> Add:
        return diff(self.H, self.x)

    @property
    def dH_dx_dx(self) -> Add:
        return diff(self.dH_dx, self.x)

    @property
    def dH_dy(self) -> Add:
        return diff(self.H, self.y)

    @property
    def dH_dy_dy(self) -> Add:
        return diff(self.dH_dy, self.y)

    def get_value_at_point(self, x: Rational, y: Rational):
        return self.H.subs(self.x, x).subs(self.y, y)
