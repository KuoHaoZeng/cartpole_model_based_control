import cv2
import matplotlib
matplotlib.use('agg')
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as patches
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np


def draw_cartpole(l, theta, p,
                  width=640, height=480, scale=50.0, cart_width=0.5, cart_height=0.25,
                  canvas=None):
    """
    :param l:  length of rod
    :param theta, p: see CartpoleSim
    :param width, height: window size
    :param scale: a multiplier that transforms metric length into number of pixels
    :param cart_width, cart_height: size of cart in meters
    :return: a uint8 BGR image of shape (height, width, 3)
    """
    if canvas is None:
        canvas = np.ones((height, width, 3), np.uint8) * 255

    cart_width = int(cart_width * scale)
    cart_height = int(cart_height * scale)

    cart_pos = int(p * scale)

    cv2.rectangle(canvas,
                  (cart_pos - cart_width // 2 + width // 2, height // 2 - cart_height // 2),
                  (cart_pos + cart_width // 2 + width // 2, height // 2 + cart_height // 2), (0, 0, 0), thickness=2)

    cv2.line(canvas, (cart_pos + width // 2, height // 2),
             (cart_pos + width // 2 + int(scale * l * np.cos(theta - np.pi / 2)),
              height - (height // 2 + int(scale * l * np.sin(theta - np.pi / 2)))),
             (0, 0, 255), thickness=3)

    return canvas


class CachedPlotter(object):
    def __init__(self, ax):
        self.ax = ax

        self.lines = {}
        self.patches = {}
        self.collections = {}
        self.images = {}
        self.texts = {}
        self.containers = {}

    def get_handle(self, name):
        for d in (self.lines, self.patches, self.collections, self.images, self.texts, self.containers):
            if name in d:
                return d[name]
        return None

    def set_visible(self, handle_name, visible):
        h = self.get_handle(handle_name)
        if h:
            h.set_visible(visible)
        else:
            print('warning: cannot find handle name %s' % handle_name)

    def clear(self):
        '''
        remove all graph elements
        '''
        for l in self.lines.values():
            self.ax.lines.remove(l)

        for c in self.collections.values():
            c.remove()

        for p in self.patches.values():
            p.remove()

        for _ in self.texts.values():
            _.remove()

        for _ in self.containers.values():
            _.remove()

        self.lines = {}
        self.collections = {}
        self.patches = {}
        self.texts = {}
        self.containers = {}

    def plot(self, handle_name, xs, ys, *args, **kwargs):
        if handle_name not in self.lines:
            self.lines[handle_name], = self.ax.plot(xs, ys, *args, **kwargs)
        h = self.lines[handle_name]
        h.set_xdata(xs)
        h.set_ydata(ys)
        self.ax.draw_artist(h)
        return h

    def plot_with_errorbar(self, handle_name, xs, ys, yerr, *args, **kwargs):
        if handle_name not in self.containers:
            self.containers[handle_name] = self.ax.errorbar(
                xs, ys, yerr, *args, **kwargs)

        lines, caplines, bars = self.containers[handle_name]
        lines.set_xdata(xs)
        lines.set_ydata(ys)

        yerr = np.array(yerr)

        erry_top, erry_bot = caplines

        erry_top.set_xdata(xs)
        erry_bot.set_xdata(xs)
        erry_top.set_ydata(ys + np.array(yerr))
        erry_bot.set_ydata(ys - np.array(yerr))

        bars[0].set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(xs, ys + yerr, ys - yerr)])

        self.ax.draw_artist(lines)
        self.ax.draw_artist(erry_top)
        self.ax.draw_artist(erry_bot)
        self.ax.draw_artist(bars[0])

        return self.containers[handle_name]

    def scatter(self, handle_name, xs, ys, *args, **kwargs):
        if handle_name not in self.collections:
            self.collections[handle_name] = self.ax.scatter(xs, ys, *args, **kwargs)
        h = self.collections[handle_name]
        # Use atleast_1d() to handle both sequences or scalars
        xys = np.hstack((np.atleast_1d(xs)[:, np.newaxis], np.atleast_1d(ys)[:, np.newaxis]))
        h.set_offsets(xys)
        self.ax.draw_artist(h)

    def polygon(self, handle_name, points, *args, **kwargs):
        if handle_name not in self.patches:
            p = patches.Polygon(points, *args, **kwargs)
            self.patches[handle_name] = p
            self.ax.add_patch(p)

        h = self.patches[handle_name]
        h.set_xy(points)
        self.ax.draw_artist(h)

    def arc(self, handle_name, xy, width, height, angle, theta1, theta2, **kwargs):
        if handle_name not in self.patches:
            p = patches.Arc(xy, width, height, angle, theta1, theta2, **kwargs)
            self.patches[handle_name] = p
            self.ax.add_patch(p)

        h = self.patches[handle_name]
        h.center = xy
        h.width = width
        h.height = height
        h.angle = angle
        h.theta1 = theta1
        h.theta2 = theta2
        self.ax.draw_artist(h)

    def line_collection(self, handle_name, lines, *args, **kwargs):
        if handle_name not in self.collections:
            collection = LineCollection(lines, *args, **kwargs)
            self.collections[handle_name] = collection
            self.ax.add_collection(collection)

        h = self.collections[handle_name]
        h.set_verts(lines)
        self.ax.draw_artist(h)

    def patch_collection(self, handle_name, patches, *args, **kwargs):
        if handle_name not in self.collections:
            collection = PatchCollection(patches, *args, **kwargs)
            self.collections[handle_name] = collection
            self.ax.add_collection(collection)

        h = self.collections[handle_name]
        h.set_paths(patches)
        self.ax.draw_artist(h)

    def image(self, handle_name, A, **kwargs):
        if handle_name not in self.images:
            im = self.ax.imshow(A, **kwargs)
            self.images[handle_name] = im
        h = self.images[handle_name]
        h.set_data(A)
        self.ax.draw_artist(h)

    def fixed_arrow(self, handle_name, x, y, dx, dy, **kwargs):
        if handle_name not in self.patches:
            p = patches.FancyArrow(x, y, dx, dy, **kwargs)
            self.patches[handle_name] = p
            self.ax.add_patch(p)

        h = self.patches[handle_name]
        self.ax.draw_artist(h)

    def fixed_arrow2(self, handle_name, x1, y1, x2, y2, **kwargs):
        if handle_name not in self.patches:
            p = patches.FancyArrowPatch((x1, y1), (x2, y2), **kwargs)
            self.patches[handle_name] = p
            self.ax.add_patch(p)
        h = self.patches[handle_name]
        self.ax.draw_artist(h)

    def text(self, handle_name, x, y, text, *args, **kwargs):
        if handle_name not in self.texts:
            t = matplotlib.text.Text(x, y, text, *args, **kwargs)
            self.texts[handle_name] = t
            self.ax.add_artist(t)
        h = self.texts[handle_name]
        h.set_text(text)
        self.ax.draw_artist(h)

    def rectangle(self, handle_name, xy, width, height, *args, **kwargs):
        if handle_name not in self.patches:
            p = patches.Rectangle(xy, width, height, *args, **kwargs)
            self.patches[handle_name] = p
            self.ax.add_patch(p)
        h = self.patches[handle_name]
        h.set_xy(xy)
        h.set_width(width)
        h.set_height(height)
        self.ax.draw_artist(h)


def _pad_width_center(w, target_w):
    left = (target_w - w) // 2
    right = target_w - w - left
    return left, right


def _pad_width_right(w, target_w):
    return 0, target_w - w


def _pad_height_center(h, target_h):
    top = (target_h - h) // 2
    bottom = target_h - h - top
    return top, bottom


def _pad_height_bottom(h, target_h):
    return 0, target_h - h


def VStack(*imgs, align='center'):
    max_w = max([_.shape[1] for _ in imgs])
    imgs_padded = []

    if align == 'center':
        for img in imgs:
            left, right = _pad_width_center(img.shape[1], max_w)
            imgs_padded.append(cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT))

    elif align == 'left':
        for img in imgs:
            left, right = _pad_width_right(img.shape[1], max_w)
            imgs_padded.append(cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT))

    else:
        raise ValueError('Unsupported alignment %s' % align)

    return np.concatenate(imgs_padded, axis=0)


def HStack(*imgs, align='center'):
    max_h = max([_.shape[0] for _ in imgs])

    imgs_padded = []

    if align == 'center':
        for img in imgs:
            top, bottom = _pad_height_center(img.shape[0], max_h)
            imgs_padded.append(cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT))

    elif align == 'top':
        for img in imgs:
            top, bottom = _pad_height_bottom(img.shape[0], max_h)
            imgs_padded.append(cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT))

    else:
        raise ValueError('Unsupported alignment %s' % align)

    return np.concatenate(imgs_padded, axis=1)


class CartpoleVisualizer(object):
    def __init__(self, cart_width=1.0, cart_height=0.5, pole_length=1.0, pole_thickness=3.0, figsize=(4, 3), title=''):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, tight_layout=True)
        self.fig = fig
        self.figsize = figsize
        self.ax = ax
        self.plotter = CachedPlotter(ax)
        self.cartpoles = dict()

        self.ax.set_xlim(-6.4, 6.4)
        self.ax.set_ylim(-4.8, 4.8)

        self.cart_width = cart_width
        self.cart_height = cart_height
        self.pole_length = pole_length
        self.pole_thickness = pole_thickness

        self.plotter.ax.get_xaxis().set_visible(False)
        self.plotter.ax.get_yaxis().set_visible(False)

        self.ax.set_title(title, fontweight='bold')
        self.fig.canvas.draw()
        self.plotter.line_collection('ground',
                                     [[(-100, -self.cart_height * 0.55), (100, -self.cart_height * 0.55)]],
                                     colors=(0.1, 0.1, 0.1), linewidths=[2.0])
        self.background = fig.canvas.copy_from_bbox(ax.bbox)

    def _draw_cartpole_helper(self, name, x, theta, alpha):
        xy = (x - self.cart_width * 0.5, 0.0 - self.cart_height * 0.5)
        self.plotter.rectangle('%s-cart' % name, xy, self.cart_width, self.cart_height, alpha=alpha)
        x2 = x + self.pole_length * np.cos(theta - np.pi * 0.5)
        y2 = self.pole_length * np.sin(theta - np.pi * 0.5)
        self.plotter.line_collection('%s-pole' % name, [[(x, 0), (x2, y2)]], linewidths=[self.pole_thickness], alpha=alpha)

    def _get_image(self):
        self.fig.canvas.blit()
        canvas = self.fig.canvas
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 3))
        return buf

    def draw_cartpole(self, name, x, theta, alpha=1.0):
        self.fig.canvas.restore_region(self.background)
        self._draw_cartpole_helper(name, x, theta, alpha)
        return self._get_image()

    def draw_cartpole_batch(self, names, xs, thetas, alphas):
        self.fig.canvas.restore_region(self.background)
        for name, x, theta, alpha in zip(names, xs, thetas, alphas):
            self._draw_cartpole_helper(name, x, theta, alpha)
        return self._get_image()


class InfoPanel(object):
    def __init__(self, figsize):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, tight_layout=True)
        self.fig = fig
        self.ax = ax
        self.plotter = CachedPlotter(ax)

        self.plotter.ax.get_xaxis().set_visible(False)
        self.plotter.ax.get_yaxis().set_visible(False)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis('off')

        fig.canvas.draw()
        self.background = fig.canvas.copy_from_bbox(ax.bbox)

    def _get_image(self):
        self.fig.canvas.blit()
        canvas = self.fig.canvas
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 3))
        return buf

    def draw_text(self, name, x, y, text, *args, **kwargs):
        self.fig.canvas.restore_region(self.background)
        self.plotter.text(name, x, y, text, *args, **kwargs)
        return self._get_image()


class Visualizer(object):
    def __init__(self, cartpole_length=1.0, x_lim=(0.0, 1.0), figsize=(6, 8)):
        fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=figsize, tight_layout=True)

        axs[0].set_title('dx', fontweight='bold')
        axs[1].set_title('ddx', fontweight='bold')
        axs[2].set_title('dtheta', fontweight='bold')
        axs[3].set_title('ddtheta', fontweight='bold')

        axs[0].set_ylim([-1, 1])
        axs[1].set_ylim([-2.5, 2.5])
        axs[2].set_ylim([-2, 2])
        axs[3].set_ylim([-10, 10])

        for ax in axs:
            ax.set_xlim(x_lim)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(True)
        axs[3].set_xlabel('time (s)')

        fig.canvas.draw()

        self.fig = fig
        self.axs = axs
        self.plotters = [CachedPlotter(_) for _ in axs]
        self.legend = None

        self.cartpole_states = dict()
        self.delta_state_trajs = dict()
        self.control = None
        self.info_text = ''
        self.cartpole_length = cartpole_length

        self.cartpole_vis_gt = CartpoleVisualizer(
            pole_length=cartpole_length, title='Full-Dynamics Prediction', figsize=(4, 3))
        self.cartpole_vis_gp = CartpoleVisualizer(
            pole_length=cartpole_length, title='GP-Dynamics Prediction', figsize=(4, 3))

        self.info_panel = InfoPanel(figsize=(4, 2))

    def _get_plot_image(self):
        canvas = self.fig.canvas
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 3))
        return buf

    def clear(self):
        for _ in self.plotters:
            _.clear()
        if self.legend is not None:
            self.legend.remove()
            self.legend = None
        self.fig.canvas.draw()

    def set_gt_cartpole_state(self, x, theta):
        self.cartpole_states['gt'] = (x, theta)

    def set_gp_cartpole_state(self, x, theta):
        self.cartpole_states['gp'] = (x, theta)

    def set_gp_cartpole_rollout_state(self, xs, thetas):
        self.cartpole_states['gp-rollout'] = (xs, thetas)

    def set_gt_delta_state_trajectory(self, ts, traj):
        self.delta_state_trajs['gt'] = np.concatenate([ts[:, None], traj], axis=1)

    def set_gp_delta_state_trajectory(self, ts, traj, variances):
        self.delta_state_trajs['gp'] = np.concatenate([ts[:, None], traj, variances], axis=1)

    def set_control(self, u):
        self.control = u

    def set_info_text(self, text):
        self.info_text = text

    def draw(self, redraw=False):
        x, theta = self.cartpole_states['gt']
        gt_cartpole = cv2.cvtColor(self.cartpole_vis_gt.draw_cartpole('', x, theta), cv2.COLOR_RGB2BGR)

        x, theta = self.cartpole_states['gp']
        xs, thetas = self.cartpole_states['gp-rollout']
        xs = np.append(xs, x)
        thetas = np.append(thetas, theta)

        names = ['gp-%d' % _ for _ in range(len(xs))]
        alphas = [0.3 for _ in range(len(xs))]
        alphas[-1] = 1.0

        gp_cartpole = cv2.cvtColor(self.cartpole_vis_gp.draw_cartpole_batch(names, xs, thetas, alphas),
                                   cv2.COLOR_RGB2BGR)

        ts, ddthetas, ddxs, dthetas, dxs = zip(*self.delta_state_trajs['gt'])

        gt_handles = []
        for idx, data in enumerate((dxs, ddxs, dthetas, ddthetas)):
            gt_handles.append(self.plotters[idx].plot('gt', ts, data, 'g', linewidth=1, alpha=0.8))

        ts, ddthetas, ddxs, dthetas, dxs, ddtheta_var, ddx_var, dtheta_var, dx_var = zip(*self.delta_state_trajs['gp'])
        vars = dx_var, ddx_var, dtheta_var, ddtheta_var
        gp_handles = []
        for idx, data in enumerate((dxs, ddxs, dthetas, ddthetas)):
            gp_handles.append(self.plotters[idx].plot_with_errorbar('gp', ts, data, np.sqrt(vars[idx]) * 3,
                              color='r', capsize=3, capthick=0.5, elinewidth=1, linewidth=1, alpha=0.8))

        if redraw:
            self.legend = self.axs[-1].legend([gt_handles[-1], gp_handles[-1]], ['Full-Dynamics', 'GP-Dynamics'],
                                             loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=2, fancybox=False)
            self.fig.canvas.draw()
        else:
            self.fig.canvas.blit()

        plot_img = cv2.cvtColor(self._get_plot_image(), cv2.COLOR_RGB2BGR)
        text_info = self.info_panel.draw_text('info', 0.0, 1.0, self.info_text, fontsize=12, verticalalignment='top')
        vis_img = HStack(VStack(gt_cartpole, gp_cartpole, text_info), plot_img)

        return vis_img


if __name__ == '__main__':
    vis = CartpoleVisualizer()
    for i in range(100):
        cv2.imshow("win", vis.draw_cartpole('test', i * 0.05, 0.0, 0.5))
        cv2.waitKey(30)
