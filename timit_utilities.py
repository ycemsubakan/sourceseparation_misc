from numpy import *
import scipy.io.wavfile as wavfile
import os
import pdb 
import librosa as lr

eps = finfo( float32).eps;

# Load a TIMIT data set
def tset( mf = None, ff = None, dr = None):
    # Where the files are
    #p = '/usr/local/timit/timit-wav/train/'
    home = os.path.expanduser('~')
    p = os.path.join(home, 'Dropbox', 'RNNs', 'timit', 'timit-wav', 'train') 

    # Pick a speaker directory
    if dr is None:
        dr = random.randint( 1, 8)
    p += '/dr%d/' % dr

    # Get two random speakers
    if mf is None:
        mf = [name for name in os.listdir( p) if name[0] == 'm']
        mf = random.choice( mf)
    if ff is None:
        ff = [name for name in os.listdir( p) if name[0] == 'f']
        ff = random.choice( ff)
    print ('dr%d/' % dr, mf, ff)

    # Load all the wav files
    #ms = [wavfile.read(p+mf+'/'+n)[1] for n in os.listdir( p+mf) if 'wav' in n]
    #fs = [wavfile.read(p+ff+'/'+n)[1] for n in os.listdir( p+ff) if 'wav' in n]

    ms = [lr.core.load(p+mf+'/'+n)[0] for n in os.listdir( p+mf) if 'wav' in n]
    fs = [lr.core.load(p+ff+'/'+n)[0] for n in os.listdir( p+ff) if 'wav' in n]

    #fs = [lr.core.load(p+mf+'/'+n)[0] for n in os.listdir( p+mf) if 'wav' in n]


    # Find suitable test file pair
    l1 = list( map( lambda x : x.shape[0], ms))
    l2 = list( map( lambda x : x.shape[0], fs))

    d = array( [[abs(t1-t2) for t1 in l1] for t2 in l2])
    i = argmin( d)
    l = max( [l1[i%10], l2[int(i/10)]])
    ts = [pad( ms[i%10], (0,l-l1[i%10]), 'constant'), pad( fs[int(i/10)], (0,l-l2[int(i/10)]), 'constant')]

    # Get training data
    ms.pop( i%10)
    fs.pop( int(i/10))
    tr = [concatenate(ms), concatenate(fs)]

    #return list(map( lambda x : (x-mean(x))/std(x), ts)), list(map( lambda x : (x-mean(x))/std(x), tr)),mf,ff
    return ts, tr, mf, ff


def sound_set( tp):
    import scipy.io.wavfile as wavfile

    # Two sinusoids signal
    if tp == 1:
        l = 8*1024
        def clip0( x):
            return x * (x>0)
        z1 = clip0( sin( 3*linspace( 0, 2*pi, l))) * sin( 1099*linspace( 0, 2*pi, l))
        z2 = clip0( sin( 2*linspace( 0, 2*pi, l))) * sin( 3222*linspace( 0, 2*pi, l))
        z3 = clip0( sin( 5*linspace( 0, 2*pi, l))) * sin( 1099*linspace( 0, 2*pi, l))
        z4 = clip0( sin( 3*linspace( 0, 2*pi, l))) * sin( 3222*linspace( 0, 2*pi, l))

        z1 = hstack( (zeros(l/8),z1))
        z2 = hstack( (zeros(l/8),z2))
        z3 = hstack( (zeros(l/8),z3))
        z4 = hstack( (zeros(l/8),z4))

    # Small TIMIT/chimes set
    elif tp == 2:
        sr,z1 = wavfile.read('/usr/local/timit/timit-wav/train/dr1/mdac0/sa1.wav')
        sr,z2 = wavfile.read('/Users/paris/Dropbox/chimes.wav')
        sr,z3 = wavfile.read('/usr/local/timit/timit-wav/train/dr1/mdac0/sa2.wav')
        z4 = z2[z1.shape[0]:]

        l = min( [z1.shape[0], z2.shape[0], z3.shape[0], z4.shape[0]])
        z1 = z1[:int(2048*floor(l/2048))]
        z2 = z2[:z1.shape[0]]
        z3 = z3[:z1.shape[0]]
        z4 = z4[:z1.shape[0]]
        z1 = z1 / std( z1)
        z2 = z2 / std( z2)
        z3 = z3 / std( z3)
        z4 = z4 / std( z4)

    # TIMIT male/female set
    elif tp == 3:
        # ts,tr = tset( 'fbjl0', 'mwsh0', 5)
        # ts,tr = tset( 'falr0', 'mtqc0', 4)
        ts,tr,mf,ff = tset()
        sr = 16000

        tr[0] = tr[0][:min(tr[0].shape[0],tr[1].shape[0])]
        tr[1] = tr[1][:min(tr[0].shape[0],tr[1].shape[0])]

        z1 = tr[1] / std( tr[1])
        z2 = tr[0] / std( tr[0])
        z3 = ts[1] / std( ts[1])
        z4 = ts[0] / std( ts[0])

    # Pad them
    sz = 1024

    def zp( x):
        return hstack( (zeros(sz),x[:int(sz*floor(x.shape[0]/sz))],zeros(sz)))

    tr1 = zp( z1[:int(sz*floor(z1.shape[0]/sz))])
    tr2 = zp( z2[:int(sz*floor(z2.shape[0]/sz))])
    ts1 = zp( z3[:int(sz*floor(z3.shape[0]/sz))])
    ts2 = zp( z4[:int(sz*floor(z4.shape[0]/sz))])

    # Show me
    #soundsc( ts1+ts2, sr)

    return tr1,tr2,ts1,ts2,mf,ff



class sound_feats:

    # Initializer
    def __init__(self, sz, hp, wn):
        import scipy.fftpack

        self.sz = sz
        self.hp = hp
        self.wn = wn

        # Forward transform definition
        self.F = scipy.fftpack.fft( identity( self.sz))

        # Inverse transform with a window
        self.iF = conj( self.wn * self.F.T)

    # Modulator definition
    def md( self, x):
        return abs( x)+eps

    # Buffer with overlap
    def buff( self, s):
        return array( [s[int(i):int(i)+self.sz] for i in arange( 0, len(s)-self.sz+1, self.hp)]).T

    # Define overlap add matrix
    def oam( self, n):
        import scipy.sparse
        ii = array( [i*self.hp+arange( self.sz) for i in arange( n)]).flatten()
        jj = array( [i*self.sz+arange( self.sz) for i in arange( n)]).flatten()
        return scipy.sparse.coo_matrix( (ones( len( ii)), (ii,jj)) ).tocsr()

    # Front end
    def fe( self, s):
        C = self.F.dot( self.wn*self.buff( s))[:int(self.sz/2)+1,:]
        M = self.md( C)
        P = C / M
        return (M,P)

    # Inverse transform
    def ife( self, M, P):
        oa = self.oam( M.shape[1])
        f = vstack( (M*P,conj(M*P)[-2:0:-1,:]))
        return oa.dot( reshape( real( self.iF.dot( f)), (-1,1), order='F')).flatten()

def bss_eval( sep, i, sources):
    # Current target
    target = sources[i]

    # Target contribution
    s_target = target * dot( target, sep.T) / dot( target, target.T)

    # Interference contribution
    pse = dot( dot( sources, sep.T), \
    linalg.inv( dot( sources, sources.T))).T.dot( sources)
    e_interf = pse - s_target

    # Artifact contribution
    e_artif= sep - pse;

    # Interference + artifacts contribution
    e_total = e_interf + e_artif;

    # Computation of the log energy ratios
    sdr = 10*log10( sum( s_target**2) / sum( e_total**2));
    sir = 10*log10( sum( s_target**2) / sum( e_interf**2));
    sar = 10*log10( sum( (s_target + e_interf)**2) / sum( e_artif**2));

    # Done!
    return (sdr, sir, sar)


