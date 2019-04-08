import math

float_inf = float('inf')
pi = math.pi
tau = 2.0 * pi
sqrt = math.sqrt

ray_eps = 0.0005
ray_inf = float_inf

# sign: -1 if v is negative; +1 if v is positive; 0 if v is zero
def sign(v): return -1 if v<0 else 1 if v>0 else 0
# clamp: returns the closest value to v that is between minv and maxv
def clamp(v, minv, maxv): return max(minv, min(maxv, v))
# sqr: returns v^2
def sqr(v): return v*v

# convert degrees to radians
def radians(deg): return deg * pi / 180.0
# convert radians to degrees
def degrees(rad): return rad * 180.0 / pi


class Vector:
    '''
    Generalized 3-dimensional vector (x,y,z)
    Note: not all functionality is provided, and I haven't tested
          every function to be bug-free or correct.
    '''
    
    __slots__ = ['x','y','z']
    
    def __init__(self, vals=None):
        self.set(vals)
    def __repr__(self):
        return '<Vector (%0.4f, %0.4f, %0.4f)>' % (self.x,self.y,self.z)
    
    def __len__(self): return 3
    def __iter__(self): return iter((self.x,self.y,self.z))
    
    def __add__(self, other):
        return Vector((self.x + other.x, self.y + other.y, self.z + other.z))
    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self
    def __sub__(self, other):
        return Vector((self.x - other.x, self.y - other.y, self.z - other.z))
    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self
    
    def __mul__(self, other):
        t = type(other)
        if t is float or t is int:
            return Vector((other * self.x, other * self.y, other * self.z))
        return Vector((other.x * self.x, other.y * self.y, other.z * self.z))
    def __imul__(self, other):
        t = type(other)
        if t is float or t is int:
            self.x *= other
            self.y *= other
            self.z *= other
        else:
            self.x *= other.x
            self.y *= other.y
            self.z *= other.z
        return self
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, other):
        t = type(other)
        if t is float or t is int:
            return Vector((self.x / other, self.y / other, self.z / other))
        return Vector((self.x/other.x, self.y/other.y, self.z/other.z))
    def __itruediv__(self, other):
        t = type(other)
        if t is float or t is int:
            self.x /= other
            self.y /= other
            self.z /= other
        else:
            self.x /= other.x
            self.y /= other.y
            self.z /= other.z
        return self
    
    def __neg__(self):
        return Vector((-self.x, -self.y, -self.z))
    
    def __setitem__(self, idx, v):
        if idx == 0: self.x = v
        elif idx == 1: self.y = v
        else: self.z = v
    def __getitem__(self, idx):
        if idx == 0: return self.x
        if idx == 1: return self.y
        return self.z
    
    def set(self, vals):
        if isinstance(vals, Vector):
            self.x,self.y,self.z = vals.x,vals.y,vals.z
        else:
            self.x,self.y,self.z = vals or (0.0,0.0,0.0)
        return self
    
    def normalize(self):
        lsqrd = self.x*self.x + self.y*self.y + self.z*self.z
        if lsqrd == 0: return self
        if abs(lsqrd-1) < 0.001:
            l = lsqrd
        else:
            l = sqrt(lsqrd)
        self.x /= l
        self.y /= l
        self.z /= l
        return self
    
    def normalized(self):
        return Vector(self.xyz).normalize()
    
    @property
    def xyz(self): return (self.x, self.y, self.z)
    @xyz.setter
    def xyz(self, v): self.set(v)
    
    @property
    def length(self):
        return sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    @property
    def length_squared(self):
        return self.x*self.x + self.y*self.y + self.z*self.z
    
    def as_vector(self): return Vector(self.xyz)
    def to_tuple(self): return self.xyz
    def copy(self): return Vector(self.xyz)
    
    def cross(self, other):
        sx,sy,sz = self.x,self.y,self.z
        ox,oy,oz = other.x,other.y,other.z
        cx = sy*oz-sz*oy
        cy = sz*ox-sx*oz
        cz = sx*oy-sy*ox
        return Vector((cx,cy,cz))
    
    def dot(self, other):
        sx,sy,sz = self.x,self.y,self.z
        ox,oy,oz = other.x,other.y,other.z
        return sx*ox + sy*oy + sz*oz
    
    def lerp(self, other, factor:float):
        sx,sy,sz = self.x,self.y,self.z
        ox,oy,oz = other.x,other.y,other.z
        f0,f1 = 1.0 - factor, factor
        return Vector((sx*f0+ox*f1, sy*f0+oy*f1, sz*f0+oz*f1))
    
    def reflected(self, direction):
        direction = direction.normalized()
        dot = direction.dot(self)
        return Vector(-self + direction * (dot*2))


class Point(Vector):
    '''
    A class representing a point in 3D space
    '''
    
    __slots__ = ['x','y','z']
    
    def __init__(self, vals=None):
        self.set(vals)
    def __repr__(self):
        return '<Point (%0.4f, %0.4f, %0.4f)>' % (self.x,self.y,self.z)
    def __add__(self, other):
        if type(other) is Point:
            return Vector((self.x+other.x, self.y+other.y, self.z+other.z))
        return Point((self.x+other.x, self.y+other.y, self.z+other.z))
    def __radd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        if type(other) is Point:
            return Vector((self.x-other.x, self.y-other.y, self.z-other.z))
        return Point((self.x-other.x, self.y-other.y, self.z-other.z))


class Direction(Vector):
    '''
    A class representing a direction in 3D space
    '''
    
    __slots__ = ['x','y','z']
    
    def __init__(self, vals=None):
        self.set(vals)
    def __repr__(self):
        return '<Direction (%0.4f, %0.4f, %0.4f)>' % (self.x,self.y,self.z)
    def __neg__(self):
        return Direction((-self.x, -self.y, -self.z))
    def set(self, vals):
        t = type(vals)
        if t is Direction or t is Normal:
            self.x,self.y,self.z = vals.x,vals.y,vals.z
            return self
        if t is Vector:
            self.x,self.y,self.z = vals.x,vals.y,vals.z
        else:
            self.x,self.y,self.z = vals or (0.0,0.0,0.0)
        return self.normalize()


class Normal(Vector):
    '''
    A class representing a normal in 3D space
    '''
    
    __slots__ = ['x','y','z']
    
    def __init__(self, vals=None):
        self.set(vals)
    def __repr__(self):
        return '<Normal (%0.4f, %0.4f, %0.4f)>' % (self.x,self.y,self.z)
    def __neg__(self):
        return Normal((-self.x, -self.y, -self.z))
    def set(self, vals):
        t = type(vals)
        if t is Direction or t is Normal:
            self.x,self.y,self.z = vals.x,vals.y,vals.z
            return self
        if t is Vector:
            self.x,self.y,self.z = vals.x,vals.y,vals.z
        else:
            self.x,self.y,self.z = vals or (0.0,0.0,0.0)
        return self.normalize()


class Ray:
    '''
    A class representing a ray in 3D space (origin and direction)
    '''
    
    __slots__ = ['e', 'd', 'min', 'max']
    
    @staticmethod
    def from_segment(a:Point, b:Point):
        return Ray(e=a, d=Direction(b - a), max_dist=(b - a).length)
    
    def __init__(self, e:Point, d:Direction, min_dist:float=0.00005, max_dist:float=float_inf):
        self.e = Point(e)
        self.d = Direction(d)
        self.min = (self.e - self.eval(min_dist)).length
        self.max = float_inf if max_dist==float_inf else (self.e - self.eval(max_dist)).length
    
    def __repr__(self):
        return '<Ray e:(%0.4f, %0.4f, %0.4f), d:(%0.4f, %0.4f, %0.4f)>' % (self.e.x,self.e.y,self.e.z,self.d.x,self.d.y,self.d.z)
    
    def eval(self, t:float): return self.e + self.d * t
    def eval_clamped(self, t:float): return self.e + self.d * clamp(t, 0, self.max)
    
    def valid_t(self, t:float)->bool: return self.min <= t and t <= self.max



class Frame:
    '''
    A class representing a coordinate frame for 3D space (origin and
    major axes (x,y,z)).  This class provides functionality for
    transforming various geometric entities between world and local
    spaces.
    
    w2l_typed, w2l_point, w2l_vector, w2l_direction, w2l_normal, w2l_ray:
        convert given entity from world-space to local-space.
    
    l2w_typed, l2w_point, l2w_vector, l2w_direction, l2w_normal, l2w_ray:
        convert given entity from local-space to world-space.
    
    The w2l_typed and l2w_typed functions call the appropriate
    transformation function based on the type of given parameter.
    Note: there is a performance penalty here, so try calling the
    appropriate function when possible.
    '''
    
    __slots__ = ['o','x','y','z','fn_l2w_typed','fn_w2l_typed']
    
    @staticmethod
    def lookat(eye:Point, center:Point, up:Direction, flipped:bool=True):
        return Frame(o=eye, y=up, z=Direction(center-eye) * (-1 if flipped else 1))

    def __init__(self, o:Point=None, x:Direction=None, y:Direction=None, z:Direction=None):
        c = (1 if x else 0) + (1 if y else 0) + (1 if z else 0)
        if c == 0:
            # major axes
            x = Direction((1,0,0))
            y = Direction((0,1,0))
            z = Direction((0,0,1))
        elif c == 1:
            if x:
                y = Direction((-x.x + 3.14, x.y + 42, x.z - 1.61))
                z = Direction(x.cross(y))
                y = Direction(z.cross(x))
            elif y:
                x = Direction((-y.x + 3.14, y.y + 42, y.z - 1.61))
                z = Direction(x.cross(y))
                x = Direction(y.cross(z))
            else:
                x = Direction((-z.x + 3.14, z.y + 42, z.z - 1.61))
                y = Direction(-x.cross(z))
                x = Direction(y.cross(z))
        elif c >= 2:
            if x and y:
                z = Direction(x.cross(y))
                y = Direction(z.cross(x))
                x = Direction(y.cross(z))
            elif x and z:
                y = Direction(z.cross(x))
                x = Direction(y.cross(z))
                z = Direction(x.cross(y))
            else:
                x = Direction(y.cross(z))
                y = Direction(z.cross(x))
                z = Direction(z)

        self.o = o or Point()
        self.x = x
        self.y = y
        self.z = z

        self.fn_l2w_typed = {
            Point:      self.l2w_point,
            Direction:  self.l2w_direction,
            Normal:     self.l2w_normal,
            Vector:     self.l2w_vector,
            Ray:        self.l2w_ray,
        }
        self.fn_w2l_typed = {
            Point:      self.w2l_point,
            Direction:  self.w2l_direction,
            Normal:     self.w2l_normal,
            Vector:     self.w2l_vector,
            Ray:        self.w2l_ray,
        }

    def __repr__(self):
        return '<Frame o:(%0.4f, %0.4f, %0.4f), x:(%0.4f, %0.4f, %0.4f), y:(%0.4f, %0.4f, %0.4f), z:(%0.4f, %0.4f, %0.4f)>' % (
            self.o.x,self.o.y,self.o.z,
            self.x.x,self.x.y,self.x.z,
            self.y.x,self.y.y,self.y.z,
            self.z.x,self.z.y,self.z.z
            )
    
    def _dots(self, v): return (self.x.dot(v), self.y.dot(v), self.z.dot(v))
    def _mults(self, v): return self.x*v.x + self.y*v.y + self.z*v.z

    def l2w_typed(self, data):
        ''' dispatched conversion '''
        t = type(data)
        assert t in self.fn_l2w_typed, "unhandled type of data: %s (%s)" % (str(data), str(type(data)))
        return self.fn_l2w_typed[t](data)
    def w2l_typed(self, data):
        ''' dispatched conversion '''
        t = type(data)
        assert t in self.fn_w2l_typed, "unhandled type of data: %s (%s)" % (str(data), str(type(data)))
        return self.fn_w2l_typed[t](data)

    def w2l_point(self, p:Point)->Point: return Point(self._dots(p - self.o))
    def l2w_point(self, p:Point)->Point: return Point(self.o + self._mults(p))

    def w2l_vector(self, v:Vector)->Vector: return Vector(self._dots(v))
    def l2w_vector(self, v:Vector)->Vector: return Vector(self._mults(v))

    def w2l_direction(self, d:Direction)->Direction: return Direction(self._dots(d))
    def l2w_direction(self, d:Direction)->Direction: return Direction(self._mults(d))

    def w2l_normal(self, n:Normal)->Normal: return Normal(self._dots(n))
    def l2w_normal(self, n:Normal)->Normal: return Normal(self._mults(n))
    
    def l2w_ray(self, ray:Ray)->Ray:
        e = self.l2w_point(ray.e)
        d = self.l2w_direction(ray.d)
        max_dist = float_inf if ray.max==float_inf else (e - self.l2w_point(ray.eval(ray.max))).length
        return Ray(e=e, d=d, max_dist=max_dist)
    def w2l_ray(self, ray:Ray)->Ray:
        e = self.w2l_point(ray.e)
        d = self.w2l_direction(ray.d)
        max_dist = float_inf if ray.max==float_inf else (e - self.w2l_point(ray.eval(ray.max))).length
        return Ray(e=e, d=d, max_dist=max_dist)

    def rotate_about_z(self, radians:float):
        c,s = math.cos(radians),math.sin(radians)
        x,y = self.x,self.y
        self.x = +x*c + y*s
        self.y = -x*s + y*c

