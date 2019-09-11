from vanila_gan import GAN
from dcgan import DCGAN
from pix2pix import Pix2Pix
from pix2pix_patch import Pix2PixPatch
from wgan import WGAN
from cycle_gan import CycleGAN

from mrigan import MRIGAN
from mrigan02 import MRIGAN02
from mrigan03 import MRIGAN03

from mrigan01_lsgan import MRIGAN01_LSGAN
from mrigan02_lsgan import MRIGAN02_LSGAN
from mrigan03_lsgan import MRIGAN03_LSGAN

from mrigan_01 import MRIGAN_01
from mrigan_02 import MRIGAN_02


# vanilla_gan, dcgan, pix2pix, pix2pix-patch, wgan
def gan_repository(sess, flags, dataset):
    if flags.gan_model == 'vanilla_gan':
        print('Initializing Vanilla GAN...')
        return GAN(sess, flags, dataset.image_size)
    elif flags.gan_model == 'dcgan':
        print('Initializing DCGAN...')
        return DCGAN(sess, flags, dataset.image_size)
    elif flags.gan_model == 'pix2pix':
        print('Initializing pix2pix...')
        return Pix2Pix(sess, flags, dataset.image_size)
    elif flags.gan_model == 'pix2pix-patch':
        print('Initializing pix2pix-patch...')
        return Pix2PixPatch(sess, flags, dataset.image_size)
    elif flags.gan_model == 'wgan':
        print('Initializing WGAN...')
        return WGAN(sess, flags, dataset)
    elif flags.gan_model == 'cyclegan':
        print('Initializing cyclegan...')
        return CycleGAN(sess, flags, dataset.image_size, dataset())
    elif flags.gan_model == 'mrigan':
        print('Initializing mrigan...')
        return MRIGAN(sess, flags, dataset.image_size, dataset())
    elif flags.gan_model == 'mrigan02':
        print('Initializing mrigan02...')
        return MRIGAN02(sess, flags, dataset.image_size, dataset())
    elif flags.gan_model == 'mrigan03':
        print('Initializing mrigan03...')
        return MRIGAN03(sess, flags, dataset.image_size, dataset())
    elif flags.gan_model == 'mrigan01_lsgan':
        print('Initializing mrigan01_lsgan...')
        return MRIGAN01_LSGAN(sess, flags, dataset.image_size, dataset())
    elif flags.gan_model == 'mrigan02_lsgan':
        print('Initializing mrigan02_lsgan...')
        return MRIGAN02_LSGAN(sess, flags, dataset.image_size, dataset())
    elif flags.gan_model == 'mrigan03_lsgan':
        print('Initializing mrigan03_lsgan...')
        return MRIGAN03_LSGAN(sess, flags, dataset.image_size, dataset())
    elif flags.gan_model == 'mrigan_01':
        print('Initializing mrigan_01...')
        return MRIGAN_01(sess, flags, dataset.image_size, dataset())
    elif flags.gan_model == 'mrigan_02':
        print('Initializing mrigan_02...')
        return MRIGAN_02(sess, flags, dataset.image_size, dataset())
    else:
        raise NotImplementedError
