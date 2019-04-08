import os
import sys

from common.utils import put_your_code_here, timed_call
from common.maths import Vector, Point, Normal, Ray, Direction, Frame, sqrt
from common.scene import Scene, Material, scene_from_file, Light
from common.image import Image

'''
The following functions provide algorithms for raytracing a scene.

`raytrace` renders an image for given scene by calling `irradiance`.
`irradiance` computes irradiance from scene along ray (reversed).
`intersect` computes intersection details between ray and scene.

You will provide your implementation to each of the functions below.
We have left pseudocode in comments for all problems but extra credits.
For extra credit, you will need to modify this and `scene.py` files.

Note: The code structure here is only a suggestion.  Feel free to
modify it as you see fit.  Your final image must match _perfectly_
the reference images.  If your final image is not **exactly** the
same, you will lose points.

Hint: Use the image to debug your code.  Store x,y,z values of ray
directions to make that they are pointing in correct direction. Store
x,y,z values of intersection points to make sure they seem reasonable.
Store x,y,z values of normals at intersections to make sure they are
reasonable.  Etc.

Hint: Implement in stages.  Trying to write the entire raytracing
system at once will often introduce multiple errors that will be
very tricky to debug.  Use the critical thinking skill of Developing
Sub-Goals to attack this project!

Remove the `@put_your_code_here` function decorators to remove the
`>>> WARNING <<<` reports when running your code.
'''


class Intersection:
    '''
    Stores information about the point of intersection:
        ray_t: t value along ray to intersection (evaluating ray at t will give pos)
        frame: shading frame of intersection (o: point of intersection, z: normal at intersection)
        mat:   the material of surface that was intersected
    '''
    
    def __init__(self, ray_t:float, frame:Frame, mat:Material):
        self.ray_t = ray_t
        self.frame = frame
        self.mat   = mat


def intersect(scene:Scene, ray:Ray):
    ''' returns shading frame at intersection of ray with scene; otherwise returns None '''
    intersection = None
    for surf in scene.surfaces:
        if surf.is_quad or surf.is_circle:
            d = ray.d
            n = surf.frame.z
            denom = d.dot(n)
            if denom == 0:
                continue
            numer = (surf.frame.o - ray.e).dot(n)
            t = numer / denom
            if not ray.valid_t(t): # check if t between tmax and tmin
                continue
            frame = surf.frame
            p = ray.eval(t)
            local_p = frame.w2l_point(p)
            x_dist = abs(local_p.x)
            y_dist = abs(local_p.y)
            if surf.is_quad and (x_dist >= surf.radius or y_dist >= surf.radius): # check if in square
                continue
            if surf.is_circle and (x_dist**2 + y_dist**2 > surf.radius**2): # check if in circle
                continue
            if intersection is None or t <= intersection.ray_t:
                i_frame = Frame(p, frame.x, frame.y, frame.z)
                mat = surf.material
                intersection = Intersection(t, i_frame, mat)
        else:
            sphere = surf
            b = 2 * ray.d.dot(ray.e - sphere.frame.o)
            c = (ray.e - sphere.frame.o).length_squared - sphere.radius * sphere.radius
            d = b*b - 4*c
            if d < 0:
                continue
            t = .5 * (-b - sqrt(d))
            if not ray.valid_t(t): # check if t between tmax and tmin
                continue
            if intersection is None or t <= intersection.ray_t:
                p = ray.eval(t)
                z_ray = ray.from_through(surf.frame.o, p)
                frame = Frame(p, None, None, z_ray.d)
                mat = sphere.material
                intersection = Intersection(t, frame, mat)


    '''
    foreach surface
        if surface is a quad
            compute ray intersection (and ray_t), continue if not hit
            check if computed ray_t is between min and max, continue if not
            check if this is closest intersection, continue if not
            record hit information
        if surface is a sphere
            compute ray intersection (and ray_t), continue if not hit
            check if computed ray_t is between min and max, continue if not
            check if this is closest intersection, continue if not
            record hit information
    return closest intersection
    '''
    
    return intersection

def irradiance(scene:Scene, ray:Ray, call_depth=0):
    ''' computes irradiance (color) from scene along ray (reversed) '''
    backColor = scene.background
    intersection = intersect(scene, ray)
    if not intersection:
        return backColor

    ambient = scene.ambient
    kd = intersection.mat.kd
    ks = intersection.mat.ks
    kr = intersection.mat.kr
    # final_color = Vector((0,0,0))
    final_color = ambient * kd
    for light in scene.lights:
        if light.is_point:
            p = intersection.frame.o
            s = light.frame.o
            vec_to_light = s-p
            ray_to_light = Ray.from_segment(p, s)
            shadow_intersection = intersect(scene, ray_to_light)
            if shadow_intersection:
                # l_dist = (s-p).length_squared
                # o_dist = (ray_to_light.eval(shadow_intersection.ray_t) - p).length_squared
                # if l_dist < o_dist:
                continue
            # light response
            l_response = light.intensity / vec_to_light.length_squared
            # light direction
            l_direction = vec_to_light.normalize()
            n = intersection.frame.z
            h = ray_to_light.d + (-ray.d)
            h = h.normalize()
            mat_n = intersection.mat.n
            temp = kd + (ks * max(0, n.dot(h))**mat_n) # ^n inside or outside
            final_color += l_response * temp * max(n.dot(l_direction), 0)
                # final_color += l_response * temp * abs(n.dot(l_direction))
        else: # directional light
            print('not yet implemented!')
    do_reflection = call_depth < 5 and not (kr.x == 0 and kr.y == 0 and kr.z == 0)
    if do_reflection:
        refl_v = -ray.d
        refl_n = intersection.frame.z
        refl_dir = -refl_v + (2*refl_v.dot(refl_n))*refl_n
        # refl_ray = Ray(e=intersection.frame.o, d=refl_dir)
        refl_ray = Ray(e=intersection.frame.o, d=refl_dir)
        final_color += kr * irradiance(scene, refl_ray, call_depth + 1)
    return final_color
 
    '''
    get scene intersection
    if not hit, return background
    accumulate color starting with ambient
    foreach light
        compute light response
        compute light direction
        compute material response (BRDF*cos)
        check for shadows and accumulate if needed
    if material has reflections
        create reflection ray
        accumulate reflected light (recursive call) scaled by material reflection
    return accumulated color
    '''
def generateRay(scene:Scene, u, v):
    o = scene.camera.frame.o
    x = scene.camera.frame.x
    y = scene.camera.frame.y
    z = scene.camera.frame.z
    w = scene.camera.width
    h = scene.camera.height
    d = scene.camera.dist
    q = o + (u-0.5) * w * x + (v-0.5) * h * y - d * z
    return Ray.from_through(o, q)


def getRayForPixelCenter(scene:Scene, row, col):
    # compute u,v
    u = (col + .5) / scene.resolution_width # [0, 1]
    v = 1 - (row + .5) / scene.resolution_height # [0, 1]
    # compute q (pt on viewing plane) 
    return generateRay(scene, u,v)

@timed_call('raytrace') # <= reports how long this function took
def raytrace(scene:Scene):
    ''' computes image of scene using raytracing '''
    
    image = Image(scene.resolution_width, scene.resolution_height)
    
    numSamples = scene.pixel_samples
    antialias = numSamples > 1
    if not antialias:
        for row in range(scene.resolution_height):
            for col in range(scene.resolution_width):
                r = getRayForPixelCenter(scene, row, col)
                image[col, row] = irradiance(scene, r)
    else: # antialiasing
        for row in range(scene.resolution_height):
            for col in range(scene.resolution_width):
                color = Vector((0,0,0))
                for row_part in range(numSamples):
                    for col_part in range(numSamples):
                        u = (col + (col_part+.5)/numSamples) / (scene.resolution_width)
                        v = 1 - (row + (row_part+.5)/numSamples) / (scene.resolution_height)
                        r = generateRay(scene, u, v)
                        color += irradiance(scene, r)
                image[col, row] = color / (numSamples**2)


    '''
    if no anti-aliasing
        foreach image row (scene.resolution_height)
            foreach pixel in row (scene.resolution_width)
                compute ray-camera parameters (u,v) for pixel
                compute camera ray
                set pixel to color raytraced with ray
    else
        foreach image row
            foreach pixel in row
                init accumulated color
                foreach sample in y
                    foreach sample in x
                        compute ray-camera parameters
                        computer camera ray
                        accumulate color raytraced with ray
                set pixel to accum color scaled by number of samples
    return rendered image
    '''
    
    return image


if len(sys.argv) < 2:
    print('Usage: %s <path/to/scenefile0.json> [path/to/scenefile1.json] [...]' % sys.argv[0])
    sys.exit(1)


for scene_filename in sys.argv[1:]:
    base,_ = os.path.splitext(scene_filename)
    image_filename = '%s.png' % base
    
    print('Reading scene: %s' % scene_filename)
    print('Writing image: %s' % image_filename)
    
    print('Raytracing...')
    scene = scene_from_file(scene_filename)
    image = raytrace(scene)
    image.save(image_filename)


print('Done')
print()