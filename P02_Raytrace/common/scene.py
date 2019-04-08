import json
from .maths import Vector, Point, Direction, Normal, Frame
from .utils import show_warning

'''
The following classes are designed to store details about the scene.

The `scene_from_file` function at bottom will load the scene details
from a properly formatted JSON file.  The loading function starts with
a Scene object and recursively assigns to the properties and creates
new objects as needed.

Note: the loading function uses reflection and property types to
know how the JSON data should be assigned and which class to
instantiate.  The function will NOT create new properties or change
types of properties.

You may add new properties or create new classes.  The only caveat is
that you must assign a default value so the loading function can
correctly discover the type, and you must include the property in the
__slots__ property.

For example: a default `Material` has no reflective coefficient (`kr`).
However `kr` must be `Vector()` and not just `None` so that:
a) we can represent no reflection using JSON (no None-like types), and
b) the loading function knows what type `kr` should be when specified
   in the JSON.

Note: properties can be expressed as `self.a_property` (object
attribute) or using the `@property` and `@a_property.setter` function
attributes (see `eye`, `center`, and `up` in `Camera`).
'''


class Material:
    __slots__ = ['kd','ks','n','kr']
    def __init__(self):
        self.kd = Vector((1,1,1))   # diffuse coefficient
        self.ks = Vector((0,0,0))   # specular coefficient
        self.n  = 10                # specular exponent
        self.kr = Vector((0,0,0))   # reflective coefficient


class Surface:
    __slots__ = ['frame','radius','is_quad', 'is_circle','material']
    def __init__(self):
        self.frame    = Frame()     # frame (origin and orientation)
        self.radius   = 1.0         # size of surface
        self.is_quad  = False       # True: quad; False: sphere
        self.is_circle  = False     # True: circle; False: sphere
        self.material = Material()  # reflective properties of surface


class Light:
    __slots__ = ['frame','intensity','is_point']
    def __init__(self):
        self.frame     = Frame()            # frame (origin and orientation)
        self.intensity = Vector((1,1,1))    # light intensity in red,green,blue
        self.is_point  = True               # True: point light; False: directional


class Camera:
    __slots__ = [
        'width','height','dist','frame',
        '_eye','_center','_up'              # <--- internally set
        ]
    def __init__(self):
        self.width  = 1.0       # image plane width
        self.height = 1.0       # image plane height
        self.dist   = 1.0       # distance from camera to center of image plane
        self.frame  = Frame()   # frame (origin and orientation)
        
        self._eye = self._center = self._up = None # internal properties, set in update_frame
        # set up default eye,center,up and corresponding frame
        self.update_frame(eye=Point((0,0,1)), center=Point((0,0,0)), up=Direction((0,1,0)))
    
    def update_frame(self, eye=None, center=None, up=None):
        ''' recomputes camera's frame based on eye,center,up '''
        if eye:    self._eye = eye
        if center: self._center = center
        if up:     self._up = up
        self.frame = Frame.lookat(self._eye, self._center, self._up)
    
    ####################################################################
    # the following properties are used to wrap a call to update_frame #
    ####################################################################
    @property
    def eye(self): return self._eye
    @eye.setter
    def eye(self, eye): self.update_frame(eye=eye)
    
    @property
    def center(self): return self._center
    @center.setter
    def center(self, center): self.update_frame(center=center)
    
    @property
    def up(self): return self._up
    @up.setter
    def up(self, up): self.update_frame(up=up)


class Scene:
    __slots__ = [
        'camera', 'resolution_width', 'resolution_height', 'pixel_samples',
        'background', 'ambient', 'lights', 'surfaces'
        ]
    def __init__(self):
        self.camera = Camera()
        self.resolution_width  = 512                # image resolution in x
        self.resolution_height = 512                # image resolution in y
        self.pixel_samples     = 1                  # samples per pixels in each direction
        self.background = Vector((0.2,0.2,0.2))     # color of background (if ray hits nothing)
        self.ambient    = Vector((0.2,0.2,0.2))     # color of ambient lighting (hack)
        self.lights   = [Light()]                   # lights in scene
        self.surfaces = [Surface()]                 # surfaces in scene


def scene_from_file(filename):
    def parse(data, cls):
        # special cases
        if cls is Vector:    return Vector(data)
        if cls is Point:     return Point(data)
        if cls is Direction: return Direction(data)
        if cls is Normal:    return Normal(data)
        
        # print('Creating %s' % str(cls))
        obj = cls()
        for k,v in data.items():
            if not hasattr(obj, k):
                show_warning('Could not find attribute "%s" in "%s" to assign value "%s"' % (k, str(obj), str(v)))
                continue
            tobj = type(getattr(obj, k))
            if   tobj is int:   v = int(v)
            elif tobj is float: v = float(v)
            elif tobj is bool:  v = bool(v)
            elif tobj is list:
                ncls = type(getattr(obj, k)[0])
                v = [parse(item, ncls) for item in v]
            else:
                ncls = type(getattr(obj, k))
                v = parse(v, ncls)
            setattr(obj, k, v)
        return obj
    
    data = json.load(open(filename, 'rt'))
    return parse(data, Scene)
