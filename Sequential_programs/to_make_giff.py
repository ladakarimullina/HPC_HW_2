# -*- coding: utf-8 -*-
"""to_make_giff.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18UvhwieU7kV_cLCRaqVj-JRhveKxxaUV
"""

from PIL import Image

nframes = 15
imgs = []

for i in range(nframes):
    imgs.append(Image.open(f'new_car_{i}.ppm'))

# Save the frames as an animated GIF
imgs[0].save('car_animation.gif',
             save_all=True,
             append_images=imgs[1:],
             duration=100,  # duration between frames in milliseconds
             loop=0)  # loop=0 means the gif will loop forever