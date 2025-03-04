import numpy as np


class Boulder():
    """
    Defines a single boulder
    """
    def __init__(self, extent, h, m2pix):
        self.extent = extent
        self.height = h
        self.m2pix = m2pix

    def set_shadow(self, theta):
        top_edge = (self.extent[0][:, 0], self.extent[1][:, 0])
        L = self.height * np.tan(theta * np.pi / 180)
        npix = int(round(L / self.m2pix, 0))
        if npix == 0:
            return None
        xmeshrange = range(top_edge[1][0], top_edge[1][-1]+1)
        ymeshrange = range(top_edge[0][0] - npix - 1, top_edge[0][0])
        return np.meshgrid(ymeshrange, xmeshrange)


class ShadowScene():
    """
    Class used for initializing a simulation of shadowing effects
    on a single pixel of M3 data.

    Parameters
    ----------
    num_boulders: int
        Number of boulders to place in the scene.
    scene_size: tulpe of ints, default: (1000, 1000)
        Pixel dimensions of the simulation scene.
    """
    def __init__(self, num_boulders: int,
                 scene_size: tuple[int] = (1000, 1000)):

        self.surface = np.zeros(scene_size)
        rng = np.random.default_rng()

        boulder_heights = rng.uniform(1, 2, num_boulders)
        MAX_SIZE = 20
        ASPECT = 3
        boulder_widths = rng.integers(1, MAX_SIZE, num_boulders)

        self.boulder_list = []

        meters2pixels = 70 / scene_size[0]

        xpixels, ypixels = np.meshgrid(np.arange(0, scene_size[0]-MAX_SIZE, 1),
                                       np.arange(0,
                                                 scene_size[0]-ASPECT*MAX_SIZE,
                                                 1))

        for h, w in zip(boulder_heights, boulder_widths):
            xpos = rng.choice(xpixels[xpixels != 0].flatten())
            ypos = rng.choice(ypixels[ypixels != 0].flatten())

            xpixels[xpos:xpos + w, ypos:ypos + w] = 0
            ypixels[xpos:xpos + w, ypos:ypos + ASPECT*w] = 0

            self.surface[xpos:xpos+w, ypos:ypos+ASPECT*w] = h

            boulder_extent = np.meshgrid(range(xpos, xpos+w),
                                         range(ypos, ypos+ASPECT*w))

            self.boulder_list.append(Boulder(boulder_extent, h, meters2pixels))

        self.boulder_area_pct = np.count_nonzero(self.surface > 0) /\
            self.surface.size
        # print(np.count_nonzero(self.surface > 0))

    def illuminate(self, incidence_angle: float):
        shadowed_surface = np.zeros_like(self.surface)
        shadowed_surface[self.surface > 0] = 2
        for i in self.boulder_list:
            shadow = i.set_shadow(incidence_angle)
            # if shadow[1].max() > shadowed_surface.shape[1]:
            if shadow[0][0, 0] < 0:
                shadow[0][:, shadow[0][0, :] < 0] = 0
            # print(shadow[1].min(), shadow[1].max())
            shadowed_surface[shadow] = 1

        shadowed_surface[self.surface > 0] = 2

        total_noshadow = np.count_nonzero(shadowed_surface != 1)
        boulder_pix = np.count_nonzero(shadowed_surface == 2)
        # print(total_noshadow, boulder_pix)
        self.boulder_area_pct = boulder_pix / total_noshadow

        return shadowed_surface
