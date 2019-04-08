import os
import sys
from math import floor
from random import randint

from common.maths import sign
from common.utils import put_your_code_here
from common.image import Image, generate_image0, generate_image1

'''
The following functions provide algorithms for rendering a scene using
raster techniques.

`render_gradient` produces an image of vertical color gradient.
`render_algorithm` produces an image following a simple algorithm.
`color_over` will mix two input colors using "over" method
`color_blend` will mix two input colors using linear blending

You will provide your implementation to each of the functions below.
We have left pseudocode in comments for most problems.

NOTE: See `image.py` for requirements in using `Image` class.

Remove the `@put_your_code_here` function decorators to remove the
`>>> WARNING <<<` reports when running your code.
'''

'''
ELECTIVE / EXTRA CREDIT

- composite functions properly handle alpha in _both_ images
- other composite functions
    - multiply / divide
    - brightness
    - additive / subtractive
    - color
- modify composite to handle images of different sizes
    - add left,top parameters to position image_A
'''


def render_gradient(start, end, steps, options):
    ''' renders a vertical color gradient '''
    
    width,height = options['width'],options['height']
    
    image = Image(width, height)
    
    '''
    foreach row in image
        foreach pixel in row
            assign color to pixel so that:
                top row is `start`
                bottom row is `end`
                rows between have `steps` evenly spaced colors
    '''
    
    for y in range(height): 
        
        rs,gs,bs = start
        re,ge,be = end
        t = int((y/height)*steps)/steps

        r = (1-t)*rs + t*re
        g = (1-t)*gs + t*ge
        b = (1-t)*bs + t*be 
        
        c = (r,g,b)
        for x in range(width):
            image[x,y] = c
            
    return image


@put_your_code_here     # <==================== Remove this line
def render_algorithm(iterations, options):
    ''' renders an image following a simple algorithm (see comments) '''
    
    width,height = options['width'],options['height']
    
    # pick three random locations and colors
    corners = [
        ((randint(0, width-1), randint(0, height-1)), (1,0,0)),
        ((randint(0, width-1), randint(0, height-1)), (0,1,0)),
        ((randint(0, width-1), randint(0, height-1)), (0,0,1)),
    ]
    
    start_position = (randint(0, width-1), randint(0, height-1))
    start_color    = (0, 0, 0)
    
    image = Image(width, height)
    
    '''
    set current position and color to start position and color
    repeat iterations
        write current color value into image at current position
        choose random corner position and its corresponding color
        update current position to be average of current and random
        update current color to be average of current and random
    '''
    pos = start_position
    col = start_color  
    for i in range(iterations): 
        image[pos] = col
        rand = randint(0,len(corners)-1)
        pos = average_pos(pos,corners[rand][0])
        col = average_col(col,corners[rand][1])
    
    return image

def average_pos (pos1, pos2):

    x1,y1 = pos1
    x2,y2 = pos2

    avg = (round((x1+x2)/2),round((y1+y2)/2))

    return avg

def average_col(col1,col2):
    
    r1,g1,b1 = col1
    r2,g2,b2 = col2

    avg = ((r1+r2)/2, (g1+g2)/2, (b1+b2)/2)

    return avg


def color_over(color_top, color_bottom):
    ''' computes color of color_top over color_bottom '''
    ''' see Image Formation slides '''

    rt,gt,bt,at = color_top
    rb,gb,bb,ab = color_bottom

    r = (at*rt) + ((1-at)*rb)
    g = (at*gt) + ((1-at)*gb)
    b = (at*bt) + ((1-at)*bb)

    return (r,g,b,1)


def color_blend(color_first, color_second, factor):
    ''' computes blending color_first and color_second by factor '''
    ''' see Image Formation slides '''

    rf,gf,bf,af = color_first
    rl,gl,bl,al = color_second

    r = (factor*rf) + (1 - factor)*rl
    g = (factor*gf) + (1 - factor)*gl
    b = (factor*bf) + (1 - factor)*bl
    a = (factor*af) + (1 - factor)*al

    return (r,g,b,a)

def color_blend_function(factor):
    return lambda t,b: color_blend(t, b, factor)


def composite_image(image_A, image_B, fn_composite):
    ''' composites image_A and image_B using fn_composite '''
    
    # images must have same width,height (extra credit)
    assert image_A.width == image_B.width
    assert image_A.height == image_B.height
    
    width,height = image_A.width,image_A.height
    image_C = Image(width, height)
    for x in range(width):
        for y in range(height):
            image_C[x,y] = fn_composite(image_A[x,y], image_B[x,y])
    return image_C


options = { 'width': 512, 'height': 512 }


print('Rendering gradient, 8 steps...')
image = render_gradient((0,0,0), (1,1,1), 8, options)
image.save('P00_gradient_008.png')

print('Rendering gradient, 16 steps...')
image = render_gradient((0,0,0), (1,1,1), 16, options)
image.save('P00_gradient_016.png')

print('Rendering gradient, 256 steps...')
image = render_gradient((0,0,0), (1,1,1), 256, options)
image.save('P00_gradient_256.png')


print('Rendering algorithm...')
image = render_algorithm(100000, options)
image.save('P00_algorithm.png')


print('Generating image A...')
imgA = generate_image0()
imgA.save('P00_A.png')      # save for reference

print('Generating image B...')
imgB = generate_image1(alpha=False) # True = varies alpha (extra credit)
imgB.save('P00_B.png')      # save for reference


print('Compositing A over B...')
image = composite_image(imgA, imgB, color_over)
image.save('P00_A_over_B.png')

print('Compositing B over A...')
image = composite_image(imgB, imgA, color_over)
image.save('P00_B_over_A.png')


print('Compositing A blend B 0.00...')
image = composite_image(imgA, imgB, color_blend_function(0.00))
image.save('P00_A_blend000_B.png')

print('Compositing A blend B 0.25...')
image = composite_image(imgA, imgB, color_blend_function(0.25))
image.save('P00_A_blend025_B.png')

print('Compositing A blend B 0.50...')
image = composite_image(imgA, imgB, color_blend_function(0.50))
image.save('P00_A_blend050_B.png')

print('Compositing A blend B 0.75...')
image = composite_image(imgA, imgB, color_blend_function(0.75))
image.save('P00_A_blend075_B.png')

print('Compositing A blend B 1.00...')
image = composite_image(imgA, imgB, color_blend_function(1.00))
image.save('P00_A_blend100_B.png')


print('Done')
