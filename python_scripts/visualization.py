#!/usr/local/bin/python3

import math
from PIL import Image, ImageDraw
import re
from collections import OrderedDict

NEW_TIMESTEP_STR=r'Body data at time = [0-9]+\.[0-9]+'
FLOAT_STR=r'[0-9]+\.[0-9]+'
ASSUMED_DENSITY = 1000000000
PI=3.141592653589
IMG_DIR="images/"
INPUT_FNAME="out.txt"
OUTPUT_FNAME="bodies.gif"
BOUND_BUFFER=100
MAX_IMG_LEN=1000

def GetBoundingBox(inFile):
    images = []
    fdata = OrderedDict()
    minx = 1E9
    miny = 1E9
    maxx = -1E9
    maxy = -1E9
    max_mass = 0
    max_stddev_x = 0
    max_stddev_y = 0
    with open(inFile) as fd:
        line = fd.readline()
        while (line):
            sum_mass = 0
            wpos_x = 0
            wpos_y = 0
            stddev_x = 0
            stddev_y = 0
            if (re.match(NEW_TIMESTEP_STR, line)):
                time = re.findall(FLOAT_STR, line)[0]
                fdata[time] = list()
            else:
                break
            line = fd.readline()
            while (not re.match(NEW_TIMESTEP_STR, line)):
                if (line):
                    body_attr = dict()
                    line_sp = line.split()
                    ndim = int((len(line_sp)-1) / 2)
                    mass = float(line_sp[0])
                    pos = line_sp[1:ndim+1]
                    sum_mass += mass
                    wpos_x += float(pos[0]) * mass
                    wpos_y += float(pos[1]) * mass
                    area = mass / ASSUMED_DENSITY
                    radius = math.sqrt(area / PI)
                    l = float(pos[0]) - radius
                    r = float(pos[0]) + radius
                    u = float(pos[1]) - radius
                    d = float(pos[1]) + radius
                    if (l < minx):
                        minx = l
                    if (u < miny):
                        miny = u
                    if (r > maxx):
                        maxx = r
                    if (d > maxy):
                        maxy = d
                    if (mass > max_mass):
                        max_mass = mass
                    body_attr['mass'] = mass
                    body_attr['pos'] = pos
                    body_attr['l'] = l
                    body_attr['r'] = r
                    body_attr['u'] = u
                    body_attr['d'] = d
                    fdata[time].append(body_attr)
                    line = fd.readline()
                else:
                    break

            xcenter = wpos_x / sum_mass
            ycenter = wpos_y / sum_mass
            for body_attr in fdata[time]:
                stddev_x += math.pow((float(pos[0]) - xcenter), 2)
                stddev_y += math.pow((float(pos[1]) - ycenter), 2)
            stddev_x = math.sqrt(stddev_x)
            stddev_y = math.sqrt(stddev_y)
            if stddev_x > max_stddev_x:
                max_stddev_x = stddev_x
            if stddev_y > max_stddev_y:
                max_stddev_y = stddev_y

        offset_x = 0.3*max_stddev_x
        offset_y = 0.3*max_stddev_y
        return xcenter - offset_x, xcenter + offset_x, ycenter - offset_y, ycenter + offset_y, max_mass, fdata


def DrawSpace(data, minx, maxx, miny, maxy, max_mass):
    images = []
    scale = (max([maxx-minx, maxy-miny]) / MAX_IMG_LEN)
    size = (BOUND_BUFFER + int((maxx-minx)/scale), BOUND_BUFFER + int((maxy-miny)/scale))
    side_buffer = BOUND_BUFFER / 2
    for time, bodies in data.items():
        image = Image.new(mode="RGB", size=size)
        draw = ImageDraw.Draw(image)
        for body in bodies:
            l = (body['l'] - minx ) / scale + side_buffer - 5
            r = (body['r'] - minx ) / scale + side_buffer + 5
            u = (body['u'] - miny ) / scale + side_buffer - 5
            d = (body['d'] - miny ) / scale + side_buffer + 5
            mass = body['mass']
            color_val = int(255 * math.log(1 + (max_mass-mass)/max_mass))
            draw.ellipse([(l, u), (r, d)], fill=(255, color_val, color_val, 255))

        images.append(image)

    images[0].save(OUTPUT_FNAME, save_all=True, optimize=False, append_images=images[1:], loop=0)

if __name__ == '__main__':
    l, r, u, d, m, data = GetBoundingBox(INPUT_FNAME)
    DrawSpace(data, l, r, u, d, m)


