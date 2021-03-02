import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# read img and set gray color
img = cv2.imread('your_img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# dashboard settings
fig, (ax, ax_2) = plt.subplots(1, 2)
plt.subplots_adjust(left=0.25, bottom=0.45)

# create slider spaces
axcolor = 'lightgoldenrodyellow'
ax_sliders = [plt.axes([0.25, 0.1 + 0.05 * i, 0.65, 0.03], facecolor=axcolor) for i in range(6)]

# define parameter sliders
ksize = Slider(ax_sliders[0], 'ksize', 1, 40, valinit=21, valstep=1)
sigma = Slider(ax_sliders[1], 'sigma', 0.1, 20.0, valinit=8, valstep=0.1)
lambd = Slider(ax_sliders[2], 'lambd', 0.1, 20.0, valinit=10, valstep=0.1)
gamma = Slider(ax_sliders[3], 'gamma', 0.1, 1, valinit=0.5, valstep=0.05)
psi = Slider(ax_sliders[4], 'psi', -10, 10, valinit=0, valstep=1)
theta = Slider(ax_sliders[5], 'theta', -5, 5, valinit=0, valstep=0.1)

sliders = [ksize, sigma, lambd, gamma, psi, theta]


def update(val):
    # on slider update recalculate gabor kernel
    g_kernel = cv2.getGaborKernel(ksize=(ksize.val, ksize.val),
                                  sigma=sigma.val,
                                  theta=np.pi / 4 * theta.val,
                                  lambd=lambd.val,
                                  gamma=gamma.val,
                                  psi=psi.val,
                                  ktype=cv2.CV_32F)
    # recalculate img result
    res = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

    # show new img and gabor kernel
    ax.imshow(res, interpolation="nearest", cmap='gray')
    ax.set_title('gabor result on img', fontsize=10)
    ax_2.imshow(g_kernel, interpolation="nearest", cmap='gray')
    ax_2.set_title('g_kernel', fontsize=10)


for i in sliders:
    i.on_changed(update)

update(None)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    for slider in sliders:
        slider.reset()


button.on_clicked(reset)
plt.show()
