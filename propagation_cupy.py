import numpy as np
import cupy as cp

class Propagation:
    def __init__(self, wav_len, dx, dy):
        self.wav_len = wav_len
        self.dx = dx
        self.dy = dy

    def nearprop_conv(self, Comp1, sizex, sizey, d):
        # Transfer to GPU
        comp1_gpu = cp.asarray(Comp1)
        
        if d == 0:
            return Comp1
            
        x1, x2 = -sizex // 2, sizex // 2 - 1
        y1, y2 = -sizey // 2, sizey // 2 - 1
        
        # Grid on GPU
        Fx, Fy = cp.meshgrid(cp.arange(x1, x2+1), cp.arange(y1, y2+1))
        
        # FFT on GPU
        Fcomp1 = cp.fft.fftshift(cp.fft.fft2(comp1_gpu)) / cp.sqrt(sizex * sizey)
        
        # Phase factor on GPU
        FresR = cp.exp(-1j * cp.pi * self.wav_len * d * ((Fx**2) / ((self.dx * sizex)**2) + (Fy**2) / ((self.dy * sizey)**2)))
        Fcomp2 = Fcomp1 * FresR
        
        # IFFT on GPU
        res_gpu = cp.fft.ifft2(cp.fft.ifftshift(Fcomp2)) * cp.sqrt(sizex * sizey)
        
        # Transfer back to CPU
        return cp.asnumpy(res_gpu)
