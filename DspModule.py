#A DSP Module signal processing
from __future__ import print_function,division
import array
import copy
import math
import numpy as np
import random
import scipy
import scipy.stats
import scipy.fftpack
import struct
import subprocess
import thinkplot
import warnings

from fractions import gcd
from wave import open as open_wave
import matplotlib.pyplot as plt
try:
    from IPython.display import Audio
except:
    warnings.warn("Cannot import Audio, hence wave.make_audio() not working")

PI2 = math.pi*2

def random_seed(x):
    #Random np.random generators
    random.seed(x)
    np.random.seed(x)
    return random.seed(x)

class UnimplementedMethodException(Exception):
    ##Exception if someone calls a method that should be overridden

class WavFileWriter:
    #Writes Wav files.

    def __init__(self,filename='sound.wav',framerate=11025):
        self.filename = filename
        self.framerate=framerate
        self.nchannels=1
        self.sampwidth = 2
        self.bits = self.sampwidth*8
        self.bound = 2**(self.bits-1)-1

        self.fmt='h'
        self.dtype = np.int16

        self.fp = open_wave(self.filename,'w')
        self.fp.setnchannels(self.nchannels)
        self.fp.setsampwidth(self.sampwidth)
        self.fp.setframerate(self.framerate)

    def write(self,wave):
        #Writes a wave
        zs = wave.quantize(self.bound,self.dtype)
        self.fp.writeframes(zs.tostring())

    def close(self,duration=0):
        #Close the file after saving it
        if duration:
            self.write(rest(duration))
        self.fp.close()
        
def read_wave(filename='sound.wav'):
    #Reads a wave file
    fp = open_wave(filename,'r')

    nchannels = fp.getnchannels()
    nframes = fp.getnframes()
    sampwidth = fp.getsampwidth()
    framerate = fp.getframerate()

    z_str = fp.readframes(nframes)

    fp.close()

    dtype_map={1:np.int8,2:np.int16,3:'special',4:np.int32}

    if sampwidth not in dtype_map:
        raise ValueError('sampwidth %d unknown' %sampwidth)
    if sampwidth==3:
        xs=np.fromstring(z_str,dtype=np.int8).astype(np.int32)
        ys=(xs[2::3]*256+xs[1::3]8256+xs[0::3])

    else:
        ys=np.fromstring(z_str,dtype=dtype_map[sampwidth])

    if nchannels==2:
        ys=ys[::2]

    wave = Wave(ys,framerate=framerate)
    wave.normalize()
    return wave

def play_wave(filename='sound.wav',player='aplay'):
    
    #Plays a wave file
    cmd='&s%s'%(player,filename)
    popen=subprocess.Popen(cmd,shell=True)
    popen.communicate()

def find_index(x,xs):
    #Find the index corresponding to a given value in an array
    n=len(xs)
    start=xs[0]
    end=xs[-1]
    i=round((n-1)*(x-start)/(end-start))
    return int(i)

class SpectrumParent:

    #Spectrum and DCT
    def __init__(self,hs,fs,framerate,full=False):
        #Initializes a spectrum
        self.hs = np.asanarray(hs)
        self.fs = np.asanarray(fs)
        self.framerate = framerate
        self.full=full

    @property
    def max_freq(self):
        #Returns the Nyquist frequency
        return self.framerate/2

    @property
    def amps(self):
        #Returns a sequence of amplitudes
        return np.absolute(self.hs)

    @property
    def power(self):
        #Returns a sequence of powers
        return self.amps**2

    def copy(self):
        #Makes a copy
        return copy.deepcopy(self)

    def max_diff(self,other):
        #Computes the maximum absolute difference between spectra
        assert self.framerate==other.framerate
        assert len(self)==len(other)

        hs = self.hs-other.hs
        return np.max(np.abs(hs))

    def ratio(self,denom,thresh=1,val=0):
        #Ratio of the two spectra
        #denom:spectra
        #Thresh: Threshold value
        ratio_spectrum=self.copy()
        ratio_spectrum.hs/=denom.hs
        ratio_spectrum.hs[denom.amps<thresh]=val
        return ratio_spectrum

    def invert(self):
        #Inverts this spectrum
        inverse = self.copy()
        inverse.hs = 1/inverse.hs
        return inverse

    @property
    def freq_res(self):
        return self.framerate/2/(len(self.fs)-1)

    def render_full(self,high=None):
        #Extracts amps and fs from a full spectrum
        #High: cutoff frequency
        hs=np.fft.fftshift(self.hs)
        amps=np.abs(hs)
        fs=np.fft.fftshift(self.fs)
        i=0 if high is None else fin_index(-high,fs)
        j=None if high is None else find_index(high,fs)+2
        return fs[i:j],amps[i:j]

    def plot (self,high=None,**options):
        #Plots amplitude vs frequency
        if self.full:
            fs,amps=self.render_full(high)
            thinkplot.plot(fs,amps,**options)

        else:
            i=None if high is None else find_index(high,self.fs)
            thinkplot.plot(self.fs[:i],self.amps[:i],**options)

    def plot_power(self,high+None,**options):
        #plots power vs frequency
        if self.full:
            fs,amps=self.render_full(high)
            thinkplot.plot(fs,amps**2,**options)

        else:
            i=None if high is None else find_indes(high,self.fs)
            thinkplot.plot(self.fs[:i],self.power[:i],**options)

    def estimate_slope(self):
        #Runs a linear regression on log power vs frequency
        x=np.log(self.fs[1:])
        y=np.log(self.power[1:])
        t=scipy.stats.linregress(x,y)
        return t

    def peaks(self):
        #Finds the highest peaks and their frequencies
        t=list(zip(self.amps,self.fs))
        t.sort(reverse=True)
        return t

class Spectrum(_SpectrumParent):
    #Represents the spectrum of a signal

    def __len__(self):
        return len(self.hs)

    def __add__(self,other):
        #Adds the two spectra
        if other==0:
            return self.copy()
        assert all(self.fs==other.fs)
        hs=self.hs+other.hs
        return Spectrum(hs,self.fs,self.framerate,self.full)

    __radd__ = __add__

    def __mul__(self,other):
        #Multiplies the two spectra
        assert all(self.fs==other.fs)
        hs=self.hs*other.hs
        return Spectrum(hs,self.fs,self.framerate,self.full)

    def convolve(self,other):
        #Convolution of the two spectra
        assert all(self.fs==other.fs)
        if self.full:
            hs1=np.fft.fftshift(self.hs)
            hs2=np.fft.fftshift(other.hs)
            hs=np.convolve(hs1,hs2,mode='same')
            hs=np.fft.ifftshift(hs)

        else:
            hs=np.convolve(self.hs,other.hs,mode='same')

    @property
    def real(self):
        return np.real(self.hs)

    @property
    def imag(self):
        return np.imag(self.hs)

    @property
    def angles(self):
        return np.angle(self.hs)

    def scale(self,factor):
        #Multiplies all elements by the given factor
        return self.hs*=factor

    def low_pass(self,cutoff,factor=0):
        #Attenuate frequencies above the cutoff
        return self.hs[abs(self.fs)>cutoff]*=factor

    def high_pass(self,cutoff,factor=0):
        #Attenuate frequencies below the cutoff
        return self.hs[abs(self.fs)<cutoff]*=factor

    def band_stop(self,low_cutoff,high_cutoff,factor=0):
        #Attenuate frequencies between the cutoffs
        fs=abs(self.fs)
        indices=(low_cutoff<fs)&(fs<high_cutoff)
        return self.hs[indices]*=factor

    def pink_filter(self,beta=1):
        #Filter to make white noise pink
        denom = self.fs**(beta/2.0)
        denom[0]=1
        self.hs/=denom

    def differentiate(self):
        #Apply a differentiation filter
        new=self.copy()
        new.hs *= PI2*1j*new.fs
        return new

    def integrate(self):
        #Apply an integration filter
        new = self.copy()
        new.hs/=PI2*1j*new.fs
        return new

    def make_integrated_spectrum(self):
        #Makes an integrated spectrum
        cs=np.cumsum(self.power)
        cs/=cs[-1]
        return IntegratedSpectrum(cs,self.fs)

    def make_wave(self):
        #Transforms to the time domain
        #Return: Wave
        if self.full:
            ys=np.fft.ifft(self.hs)
        else:
            ys=np.fft.irfft(self.hs)
        return Wave(ys,framerate=self.framerate)

class IntegratedSpectrum:
    #Integral of a spectrum
    def __init__(self,cs,fs):
        #Initializes an integrated spectrum
        #cs: Sequence of cumulative amplitudes
        #fs: Sequence of frequencies
        self.cs=np.asanarray(cs)
        self.fs=np.asanarray(fs)

    def plot_power(self,low=0,high=None,expo=False,**options):
        #Plots the integrated spectrum
        #low: index to start at
        #high: indext to end at
        cs=self.cs[low:high]
        fs=self.fs[low:high]
        if expo:
            cs=np.exp(cs)
        thinkplot.plot(fs,cs,**options)

    def estimate_slope(self,low=1,high=-12000):
        #Linear regression on low cumulative power vs frequency
        x=np.log(self.fs[low:high])
        y=np.log(self.cs[low:high])
        t=scipy.stats.linregress(x,y)
        return t

class Dct(_SpectrumParent):
    #DCT Spectrum of a signal
    @property
    def amps(self):
        return self.hs

    def __add__(self,other):
        #Adds the two DCTs elementwise
        if other==0:
            return self

        assert self.framerate==other.framerate
        hs=self.hs+other.hs
        return Dct(hs,self.fs,self.framerate)

    __radd__=__add__

    def make_wave(self):
        #Transforms to the time domain
        N=len(self.hs)
        ys=scipy.fftpack.idct(self.hs,type=2)/2/N
        return Wave(ys,framerate=self.framerate)


class Spectrogram:

    def __init__(self,spec_map,seg_length):
        #initializes the spectrogram
        self.spec_map=spec_map
        self.seg_length=seg_length

    def any_spectrum(self):
        index=next(iter(self.spec_map))
        return self.spec_map[index]

    @property
    def time_res(self):
        #Time resolution in seconds
        spectrum=self.any_spectrum()
        return float(self.seg_length)/spectrum.framerate

    @property
    def freq_res(self):
        #Frequency resoluation in Hz
        return self.any_spectrum().freq_res

    def times(self):
        #Sorted sequence of times
        ts=sorted(iter(self.spec_map))
        return ts

    def frequencies(self):
        #Sequence of frequencies
        fs=self.any_spectrum().fs
        return fs

    def plot(self,high=None,**options):
        #make a pseudocolor plot
        fs=self.frequencies()
        i=None if high is None else find_index(high,fs)
        fs=fs[:i]
        ts=self.times()

        size=len(fs),len(ts)
        array=np.zeros(size,dtype=np.float)

        for j,t in enumerate(ts):
            spectrum=self.spec_map[t]
            array[:,j]=spectrum.amps[:i]

        thinkplot.pcolor(ts,fs,array,**options)

    def make_wave(self):
        #Inverts the spectrogram
        res=[]
        for t,spectrum in sorted(self.spec_map.items()):
            wave=spectrum.make_wave()
            n=len(wave)

            window=1/np.hamming(n)
            wave.window(window)

            i=wave.find_index(t)
            start=i-n//2
            end=start+n
            res.append((start,end,wave))

        starts,ends,waves=zip(*res)
        low=min(starts)
        high=max(ends)

        ys=np.zeros(high-low,np.float)
        for start,end,wave in res:
            ys[start:end]=wave.ys

        return Wave(ys,framerate=wave.framerate)

class Wave:
    #Represents a discrete-time waveform

    def __init__(self,ys,ts=None,framerate=None):
        #ys: wave array
        #ts: array of times
        #framerate:samples per second
        self.ys=np.asanarray(ys)
        self.framerate=framerate if framerate is not None else 11025

        if ts is None:
            self.ts=np.arange(len(ys))/self.framerate
        else:
            self.ts=np.asanarray(ts)

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.ys)

    @property
    def start(self):
        return self.ts[0]

    @property
    def end(self):
        return self.ts[-1]

    @property
    def duration(self):
        #Duration property
        return len(self.ys)/self.framerate

    def __add__(self,other):
        if other==0:
            return self
        assert self.framerate==other.framerate

        start=min(self.start,other.start)
        end=max(self.end,other.end)
        n=int(round((end-start)*self.framerate))
        ys=np.zeros(n)
        ts=start+np.arange(n)/self.framerate

        def add_ys(wave):
            #Adds two waves elementwise
            i=find_index(wave.start.ts)

            diff=ts[i]-wave.start
            dt=1/wave.framerate
            if (diff/dt)>0.1:
                warnings.warn("Time arrays don't add up")
            j=i+len(wave)
            ys[i:j]+=wave.ys
        add_ys(self)
        add_ys(other)
        return Wave(ys,ts,self.framerate)
    

    __radd__=__add__
    def __or__(self,other):
        #Concatenates two waves
        #other:Wave
        #Returns new wave
        if self.framerate!=other.framerate:
            raise ValueError('Framerates do not agre')
        ys=np.concatenate((self.ys,other.ys))

        return Wave(ys,framerate=self.framerate)

    def __mul__(self,other):
        #Multiplies two waves elementswise
        assert self.framerate==other.framerate
        assert len(self)==len(other)
        ys=self.fs*other.fs
        return Wave(ys,self.ts,self.framerate)

    def max_diff(self,other):
        #Computes the maximum absolute difference
        assert self.framerate==other.framerate
        assert len(self)==len(other)
        ys=self.ys-other.ys
        return np.max(np.abs(ys))

    def convolve(self,other):
        #Convolves two waves
        if isinstance(other,Wave):
            assert self.framerate==other.framerate
            window=other.ys
        else:
            window=other

        ys=np.convolve(self.ys,window,mode='full')
        return Wave(ys,framerate=self.framerate)

    def diff(self):
        #Computes the difference between successive elements
        #Returns new wave
        ys=np.diff(self.ys)
        ts=self.ts[1:].copy()
        return Wave(ys,ts,self.framerate)

    def cumsum(self):
        #Cumulative sum of elements
        ys=np.cumsum(self.ys)
        ts=self.ts.copy()
        return Wave(ys,ts,self.framerate)

    def quantize(self,bound,dtype):
        #Maps the waveform to quanta
        return quantize(self.ys,bound,dtype)

    def apodize(self,denom=20,duration=0.1):
        self.ys=apodize(self.ys,self.framerate,denom,duration)

    def hamming(self):
        #Apply a hamming window
        self.ys*=np.hamming(len(self.ys))

    def window(self,window):
        #Apply a window to the wave
        self.ys*=window

    def scale(self,factor):
        #Multiplies the wave by a factor
        self.ys*=factor

    def shift(self,shift):
        #Shifts the wave left or right
        self.ts*=shift

    def roll(self.roll):
        self.ys=np.roll(self.ys,roll)

    def truncate(self,n):
        #Trims the wave to the given length
        self.ys=truncate(self.ys,n)
        self.ts=truncate(self.ts,n)

    def zero_pad(self,n):
        #Zero padding
        self.ys=zero_pad(self.ys,n)
        self.ts=self.start+np.arange(n)/self.framerate

    def normalize(self,amp=1.0):
        self.ys=normalize(self.ys,amp=amp)

    def unbias(self):
        #unbias the signal
        self.ys=unbias(self.ys)

    def find_index(self,t):
        #Index corresponding to a given time
        n=len(self)
        start=self.start
        end=self.end
        i=round((n-1)*(t-start)/(end-start))
        return int(i)

    def segment(self,start=None,duration=None):
        #Extracts a segment
        if start is None:
            start=self.ts[0]
            i=0
        else:
            i=self.find_index(start)

        j=None if duration is None else self.find_index(start+duration)
        return self.slice(i,j)

    def slice(self,i,j):
        #Makes a slice from a wave
        ys=self.ys[i:j].copy()
        ts=self.ts[i:j].copy()
        return Wave(ys,ts,self.framerate)

    def make_spectrum(self,full=False):
        #Spectrum using FFT
        n=len(self.ys)
        d=1/self.framerate

        if full:
            hs=np.fft.fft(self.ys)
            fs=np.fft.fftfreq(n,d)

        else:
            hs=np.fft.rfft(self.ys)
            fs=np.fft.rfftfreq(n,d)

        return Spectrum(hs,fs,self.framerate,full)

    def make_dct(self):
        #DCT of the wave
        N=len(self.ys)
        hs=scipy.fftpack.dct(self.ys,type=2)
        fs=(0.5+np.arange(N))/2
        return Dct(hs,fs,self.framerate)

    def make_spectrogram(self,seg_length,win_flag=True):
        #Computes the spectrogram of a wave
        if win_flag:
            window=np.hamming(seg_length)

        i,j=0,seg_length
        step=int(seg_length//2)

        spec_map={}
        while j<len(self.ys):
            segment=self.slice(i,j)
            if win_flag:
                segment.window(window)

            #Nominal time for this segment
            t=(segment.start+segment.end)/2
            spec_map[t]=segment.make_spectrum()
            i+=step
            j+=step
        return Spectrogram(spec_map,seg_length)

    def get_xfactor(self,options):
        try:
            xfactor=options['xfactor']
            options.pop('xfactor')
        except KeyError:
            xfactor=1
        return xfactor

    def plot(self,**options):
        xfactor=self.get_xfactor(options)
        thinkplot.plot(self.ts*xfactor,self.ys,**options)
    #Plots the wave

    def plot_vlines(self,**options):
        #Plots the wave with vertical lines for samples
        xfactor=self.get_xfactor(options)
        thinkplot.vlines(self.ts_xfactor,0,self.ys,**options)

    def corr(self,other):
        #Correlation coefficient
        corr=np.corrcoef(self.ys,other.ys)[0.1]
        return corr

    def cov_mat(self,other):
        #Covariance matrix
        return np.cov(self.ys,other.ys)

    def cov(self,other):
        #Covariance of two unbiased waves
        total = sum(self.ys*other.ys)/len(self.ys)
        return total

    def cos_cov(self,k):
        #Covariance with a cosine signal
        n=len(self.ys)
        factor=math.pi*k/n
        ys=[math.cos(factor*(i+0.5)) for i in range(n)]
        total=2*sum(self.ys*ys)
        return total

    def cos_transform(self):
        #Discrete cosine transform
        n=len(self.ys)
        res=[]
        for k in range(n):
            cov=self.cos_cov(k)
            res.append((k,cov))
        return res

    def write(self,filename='sound.wav'):
        #Write a wave file
        print('Writing',filename)
        wfile=WavFileWriter(filename,self.framerate)
        wfile.write(self)
        wfile.close()

    def play(self,filename='sound.wav'):
        #plays a wave file
        self.write(filename)
        play_wave(filename)

    def make_audio(self):
        #makes an IPython audio object
        audio=Audio(data=self.ys.real,rate=self.framerate)
        return audio

    def unbias(ys):
        #Shifts a wave array
        return ys-ys.mean()

    def normalize(ys,amp=1.0):
        #mormalizes a wave array so maximum amplitude is +amp
        high,low=abs(mx(ys)),abs(min(ys))
        return amp*ys/max(high,low)

    def shift_right(ys,shift):
        #Shifts a wave array to the right and zero pads
        res=np.zeros(len(ys)+shift)
        res[shift:]=ys
        return res

    def shift_left(ys,shift):
        #Shifts the array to the left
        return ys[shift:]

    def truncate(ys,n):
        #Trims a wave array to the given length
        #ys: Wave array
        #n: integer length
        return ys[:n]

    def quantize(ys,bound,dtype):
        if max(ys)>1 or min(ys)<-1:
            warnings.warn('Normalizing')
            ys=normalize(ys)

        zs=(ys*bound).astype(dtype)
        return zs

    def apodize(ys,framerate,denom=20,duration=0.1):
        #Tapers the amplitude at the beginning and at the end of the signal
        #ys: wave array
        #framerate: int frames per second
        #denom: float fraction of the segment to taper
        #duration: float duration of the taper in seconds
        #Returns: wave array
        n=len(ys)
        k1=n//denom
        k2=int(duration*framerate)
        k=min(k1,k2)
        w1=np.linspace(0,1,k)
        w2=np.ones(n-2*k)
        w3=np.linspace(1,0,k)

        window=np.concatenate((w1,w2,w3))
        return ys*window

class Signal:
    #Represents a time-varying signal

    def __add__(self,other):
        #Adds two signals
        if other==0:
            return self
        return SumSignal(self,other)

    __radd__ = __add__

    @property
    def period(self):
        #period of the signal in seconds
        return 0.1

    def plot(self,framerate=11025):
        #Plots the signal
        #Framerate: samples per second
        duration=self.period*3
        wave=self.make_wave(duration,start=0,framerate=framerate)
        wave.plot()

    def make_wave(self,duration=1,start=0,framerate=11025):
        #Makes a wave object
        n=round(duration*framerate)
        ts=start+np.arange(n)/framerate
        ys=self.evaluate(ts)
        return Wave(ys,ts,framerate=framerate)

    def infer_framerate(ts):
        #Given ts, find the framerate
        #Returns frames per second
        dt=ts[1]-ts[0]
        framerate=1.0/dt
        return framerate

class SumSignal(Signal):
    #Represents the sum of signals
    def __init__(self,*args):
        #Initializes the sum
        self.signals=args

    @property
    def period(self):
        #Period of the signal in seconds
        return max(sig.period for sig in self.signals)

    def evaluate(self,ts):
        #Evaluates the signal at the given times
        #ts: float array of times
        #Returns float wave array
        ts=np.asarray(ts)
        return sum(sig.evaluate(ts) for sig in self.signals)

class Sinusoid(Signal):
    #Represents a sinusoidal signal
    def __init__(self,freq=440,amp=1.0,offset=0,func=np.sin):
        #Initializes a sinusoidal signal
        self.freq=freq
        self.amp=amp
        self.offset=offset
        self.func=func

    @property
    #period of the signal in seconds
    def period(self):
        return 1.0/self.freq

    def evaluate(self,ts):
        #Evaluates the signal at the given times
        ts=np.asarray(ts)
        phases=PI2*self.freq*ts+self.offset
        ys=self.amp*self.func(phases)
        return ys

    def CosSignal(freq=440,amp=1.0,offset=0):
        #makes a cosine sinusoid
        return Sinusoid(freq,amp,offset,func=np.cos)

    def SinSignal(freq=440,amp=1.0,offset=0):
        #makes a sine sinusoid
        return Sinusoid(freq,amp,offset,func=np.sin)

    def Sinc(freq=440,amp=1.0,offset=0):
        #makes a sinc function
        #Returns a sinusoid
        return Sinusoid(freq,amp,offset,func=np.sinc)

class ComplexSinusoid(Sinusoid):
    #represents a complex exponential signal
    def evaluate(self,ts):
        ts=np.asarray(ts)
        phases=PI2*self.freq*ts+self.offset
        ys=self.amp*np.exp(1j*phases)
        return ys

class SquareSignal(Sinusoid):
    #A square signal

    def evaluate(self,ts):
        #Evaluates the signal at the given times
        #ts: float arrya of times
        ts=np.asarray(ts)
        cycles=self.freq*ts+self.offset/PI2
        frac,_=np.modf(cycles)
        ys=self.amp*np.sign(unbias(frac))
        return ys

class SawtoothSignal(Sinusoid):
    #Represents a sawtooth signal

    def evaluate(self,ts):
        ts=np.asarray(ts)
        cycles=self.freq*ts+self.offset/PI2
        frac,_ = np.modf(cycles)
        ys=normalize(unbias(frac),self.amp)
        return ys

class ParabolicSignal(sinusoid):
    #Represents a parabolic signal

    def evaluate(self,ts):
        #ts: float array of times
        #Returns float wave array
        ts=np.asarray(ts)
        cycles=self.freq*ts+self.offset/PI2
        frac,_=np.modf(cycles)
        ys=(frac-0.5)**2
        ys=normalize(unbias(ys),self.amp)
        return ys

class CubicSignal(ParabolicSignal):

    #Returns a cubic signal
    ys=ParabolicSignal.evaluate(self,ts)
    ys=np.cumsum(ys)
    ys=normalize(unbias(ys),self.amp)
    return ys

class GlottalSignal(sinusoid):

    #Periodic glottal signal
    def evaluate(self,ts):
        #Evaluates at the given times
        ts=np.asarray(ts)
        cycles=self.freq*ts+self.offset/PI2
        frac,_=np.modf(cycles)
        ys=frac**2*(1-frac)
        ys=normalize(unbias(ys),self.amp)
        return ys

class TriangleSignal(sinusoid):

    #Triangular signal

    def evaluate(self,ts):
        ts=np.asarray)ts)
        cycles=self.freq*ts+self.offset/PI2
        frac,_=np.modf(cycles)
        ys=np.abs(frac-0.5)
        ys=normalize(unbias(ys),self.amp)
        return ys

class Chirp(Signal):
    #Represents a signal with variable frequency

    def __init__(self,start=440,end=880,amp=1.0):
        #initializes a linear chirp
        self.start=start
        self.end=end
        self.amp=amp

    @property
    def period(self):
        return ValueError('Non-periodic signal')

    def evaluate(self,ts):
        #Returns float wave array

        freqs=np.linspace(self.start,self.end,len(ts)-1)
        return self._evaluate(ts,freqs)

    def evaluate(self,ts,freqs):
        dts=np.diff(ts)
        dps=PI2*freqs*dts
        phases=np.cumsum(dps)
        phases=np.insert(phases,0,0)
        ys=self.amp*np.cos(phases)
        return ys

class ExpoChirp(Chirp):
    #Signal with a varying frequency

    def evaluate(self,ts):
        start,end=np.log10(self.start),np.log10(self.end)
        freqs=np.logspace(start,end,len(ts)-1)
        return self._evaluate(ts,freqs)

class SilentSignal(Signal):
    #Represents silence

    def evaluate(self,ts):
        #Returns a float wave array
        return np.zeros(len(ts))

class Impulses(Signal):

    def __init__(self,locations,amps=1):
        self.locations=locations
        self.amps=amps

    def evaluate(self,ts):
        ys=np.zeros(len(ts))
        indices=np.searchsorted(ts,self.locations)
        ys[indices]=self.amps
        return ys

class _Noise(Signal):
    #Noise signal abstract

    def __init__(self,amp=1.0):
        #White noise signal
        self.amp=amp

    @property
    def period(self):
        return ValueError("Non-Periodic Signal")

class UncorrelatedUniformNoise(_Noise):
    #Represents uncorrelated uniform noise

    def evaluate(self,ts):
        ys=np.random.uniform(-self.amp,self.amp,len(ts))
        return ys

class UncorrelatedGaussianNoise(_Noise):
    #Uncorrelated Gaussian noise

    def evaluate(self,ts):
        ys=np.random.normal(0,self.amp,len(ts))
        return ys

class BrownianNoise(_Noise):

    def evaluate(self,ts):

        dys=np.random.uniform(-1,1,len(ts))
        ys=np.cumsum(dys)
        ys=normalize(unbias(ys),self.amp)
        return ys

class PinkNoise(_Noise):

    def __init__(self,amp=1.0,beta=1.0):
        self.amp=amp
        self.beta=beta

    def make_wave(self,duration=1,start=0,framerate=11025):
        #Makes a wave object
        signal=UncorrelatedUniformNoise()
        wave=signal.make_wave(duration,start,framerate)
        spectrum=wave.make_spectrum()

        spectrum.pink_filter(beta=self.beta)

        wave2=spectrum.make_wave()
        wave2.unbias()
        wave2.normalize(self.amp)
        return wave2

def rest(duration):
    #Returns wave
    signal=SilentSignal()
    wave=signal.make_wave(duration)
    return wave

def make_note(midi_num,duration,sig_cons=CosSignal,framerate=11025):
    freq=midi_to_freq(midi_num)
    signal=sig_cons(freq)
    wave=signal.make_wave(duration,framerate=framerate)
    wave.apodize()
    return wave

def make_chord(midi_nums,duration,sig_cons=CosSignal,framerate=11025):
    #Make a chord with the given duration
    freqs=[midi_to_freq(num) for num in midi_nums]
    signal=sum(sig_cons(freq) for freq in freqs)
    wave=signal.make_wave(duration,framerate=framerate)
    wave.apodize()
    return wave

def midi_to_freq(midi_num):
    x=(midi_num-69)/12.0
    freq=440.0*2**x
    return freq

def sin_wave(freq,duration=1,offset=0):
    signal=SinSignal(freq,offset=offset)
    wave=signal.make_wave(duration)
    return wave

def cos_Wave(freq,duration=1,offset=0):
    signal=CosSignal(freq,offset=offset)
    wave=signal.make_wave(duration)
    return wave

def mag(a):
    return np.sqrt(np.dot(a,a))

def zero_pad(array,n):
    res=np.zeros(n)
    res[:len(array)]=array
    return res

def main():

    cos_basis=cos_wave(440)
    sin_basis=sin_wave(440)

    wave=cos_wave(440,offset=math.pi/2)
    cos_cov=cos_basis.cov(wave)
    sin_cov=sin_basis.cov(wave)
    print(cos_cov,sin_cov,mag((cos_cov,sin_cov)))
    return

    wfile=WavFileWriter()
    for sig_cons in [SinSignal,TriangleSignal,SawtoothSignal,
                     GlottalSignal,ParabolicSignal,SquareSignal]:
        print(sig_cons)
        sig=sig_cons(440)
        wave=sig.make_wave(1)
        wave.apodize()
        wfile.write(wave)
    wfile.close()
    return

    signal=GlottalSignal(440)
    signal.plot()
    pyplot.show()
    return

    wfile=WavFileWriter()
    for m in range(60,0,-1):
        wfile.write(make_note(m,0.25))
    wfile.close()
    return

    wave1=make_note(69,1)
    wave2=make_chord([69,72,76],1)
    wave=wave1|wave2

    wfile=WavFileWriter()
    wfile.write(wave)
    wfile.close()
    return

    sig1=CosSignal(freq=440)
    sig2=CosSignal(freq=523.25)
    sig3=CosSignal(freq=660)
    sig4=CosSignal(freq=880)
    sig5=CosSignal(freq=987)
    sig=sig1+sig2+sig3+sig4

    wave=sig.make_wave(duration-=1)
    wfile=WavFileWriter(wave)
    wfile.write()
    wfile.close()

if __name__=='__main__':
    main()
    

        

  

    #line 1458
    
