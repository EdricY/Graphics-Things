#import png     # see: https://pythonhosted.org/pypng/png.html
from math import pi, cos, sin, floor, asin
from .png import Reader, Writer
from .png import from_array as png_from_array
from .maths import clamp, sqrt

'''
The following class provides basic functionality for creating, loading,
and saving images in PNG format.

See test code at bottom for example of using `Image`.

The Image class is column-ordered, so positions are specified `(x,y)`,
and component values for red,green,blue,alpha are floats in [0,1] range.
Note: you can assign values that are outside [0,1] (ex: -1.2, 42), but
they will be clamped to [0,1] when saving.

The pixel getter and setter functions (ex: `image[x,y]`,
`image.set((x,y), (0,0,0,1))`) REQUIRE that the x and y position values
are integers within proper respective ranges: [0, width) and [0,height).
Use the `int` function to convert floats to integers (ex:
`image[int(x), int(y)]`).

The pixel setter functions can take a 3- or 4-value color tuple,
representing either red,green,blue or red,green,blue,alpha colors,
respectively.  If alpha is not given, alpha=1.

The pixel getter functions always returns a 4-tuple (RGBA).
'''


class Image:
    @staticmethod
    def from_file(filename):
        p = Reader(filename=filename)
        width,height,pixels,metadata = p.asRGBA8()
        pixels = [[v/255.0 for v in row] for row in pixels]
        return Image(width, height, pixels=pixels)
    
    def __init__(self, width, height, pixels=None, default_color=(0,0,0,1)):
        self.width,self.height = width,height
        
        # self.pixels uses "boxed row flat pixel" format
        if pixels is None:
            self.pixels = [[c for x in range(width) for c in default_color] for y in range(height)]
        else:
            self.pixels = pixels
    
    def __setitem__(self, pos, color):
        x,y = pos
        if len(color) == 3: color = [color[0], color[1], color[2], 1]
        self.pixels[y][x*4:x*4+4] = color
    
    def __getitem__(self, pos):
        x,y = pos
        return self.pixels[y][x*4:x*4+4]
    
    def set(self, pos, color):
        x,y = pos
        if len(color) == 3: color = [color[0], color[1], color[2], 1]
        self.pixels[y][x*4:x*4+4] = color
    
    def get(self, pos):
        x,y = pos
        return self.pixels[y][x*4:x*4+4]
    
    def save(self, filename):
        info = {'width':self.width, 'height':self.height, 'bitdepth':8}
        pixels = [[int(255*clamp(v,0,1)) for v in row] for row in self.pixels]
        p = png_from_array(pixels, mode="RGBA", info=info)
        p.save(filename)


def generate_image0():
    img = Image(512, 512)
    for x in range(512):
        for y in range(512):
            dx,dy = x-256,y-256
            d = sqrt(dx*dx + dy*dy)
            if d > 205:
                r,g,b,a = 0,1,0,0
            elif d > 200:
                r,g,b,a = 0,0,0,1
            elif d > 150:
                r,g,b,a = 1,1,1,1
            elif d > 145:
                r,g,b,a = 0,0,0,1
            elif d > 100:
                r,g,b,a = 0.2,0.2,0.8,0.5
            else:
                r,g,b = 1,1,1
                a = (x+y)%2
            img[x,y] = r,g,b,a
    return img

def generate_image1(alpha=False):
    img = Image(512, 512)
    for x in range(512):
        for y in range(512):
            v = abs(sin((y-256)/256 * pi/2))
            checker = (floor(x*20/512) + floor(asin((y-256)/256)*20/(pi/2))) % 2
            if checker == 0:
                # r,g,b = v,pow(1-v,2),0.1
                r,g,b = 1,0.1,0.1
            else:
                r,g,b = 0.1,1,0.1
            a = 1 if not alpha else pow(v,2)
            img[x,y] = r,g,b,a
    return img

if __name__ == "__main__":
    img = Image(256, 256)
    
    for x in range(256):
        for y in range(256):
            r = 1 - x/255
            g = y/255
            b = 0.0
            a = 0.75
            img[x,y] = r,g,b,a
    
    for x in range(256):
        for y in range(256):
            r,g,b,a = img[x,y]
            b += 0.5
            img[x,y] = r,g,b,a
    
    img.save('test_image.png')

