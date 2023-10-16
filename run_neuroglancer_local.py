import neuroglancer
import numpy as np
import imageio
import h5py

import sys


if __name__ == '__main__':

    ip = 'localhost' #or public IP of the machine for sharable display
    port = 9090 #change to an unused port number
    neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
    viewer=neuroglancer.Viewer()

    # SNEMI (# 3d vol dim: z,y,x)
    D0='./'
    res = neuroglancer.CoordinateSpace(
            names=['x', 'y'],
            units=['nm', 'nm'],
            scales=[4, 4])

    im = np.load(sys.argv[1])
    print(im.shape)

    def ngLayer(data,res,oo=[0,0],tt='segmentation'):
        return neuroglancer.LocalVolume(data,dimensions=res,volume_type=tt,voxel_offset=oo)

    with viewer.txn() as s:
        s.layers.append(name='im',layer=ngLayer(im,res,tt='image'))

    print(viewer)