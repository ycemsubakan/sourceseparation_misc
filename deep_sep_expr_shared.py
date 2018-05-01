import time
from numpy import *
#from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import pdb
import torch
import utils as ut
#import theano.sandbox.cuda 
#theano.sandbox.cuda.use('gpu1')

eps = finfo( float32).eps;

import os
import scipy.io.wavfile as wavfile
from numpy import *

# Load a TIMIT data set
def tset( mf = None, ff = None, dr = None):
    # Where the files are
    p = '/Users/svnktrm2/Dropbox/timit-wav/train/';

    # Pick a speaker directory
    if dr is None:
        dr = random.randint( 1, 8)
    p += 'dr%d/' % dr

    # Get two random speakers
    if mf is None:
        mf = [name for name in os.listdir( p) if name[0] == 'm']
        mf = random.choice( mf)
    if ff is None:
        ff = [name for name in os.listdir( p) if name[0] == 'f']
        ff = random.choice( ff)
    print ('dr%d/' % dr, mf, ff)

    # Load all the wav files
    ms = [wavfile.read(p+mf+'/'+n)[1] for n in os.listdir( p+mf) if 'wav' in n]
    fs = [wavfile.read(p+ff+'/'+n)[1] for n in os.listdir( p+ff) if 'wav' in n]

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

    return list(map( lambda x : (x-mean(x))/std(x), ts)), list(map( lambda x : (x-mean(x))/std(x), tr))

#
# Load a set of files
#

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
        ts,tr = tset( 'fbjl0', 'mwsh0', 5)
        # ts,tr = tset( 'falr0', 'mtqc0', 4)
        # ts,tr = tset()
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

    return tr1,tr2,ts1,ts2


#
# Sound feature class
#

# Sound feature class
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


#
# NMF separation
#

# Define Polya Urn function
def nmf_model( x, r, ep, b=1, sp=0):
    # Constants
    m = x.shape[0]
    n = x.shape[1]

    # Normalize input
    g = sum( x, axis=0)+eps
    z = x / g

    # Learn or fit?
    if isscalar( r):
        w = random.rand( m, r)+10
        w /= sum( w, axis=0)
        lw = True
    else:
        w = hstack( r)
        r = w.shape[1]
        lw = False

    # Init activations
    h = random.rand( r, n)+10
    h /= sum( h, axis=0)+eps

    # Start churning
    for e in arange( ep):
        # Get tentative estimate
        v = z / (w.dot( h)+eps)
        if lw:
            nw = w * v.dot( h.T)
        nh = h * w.T.dot( v)

        # Sparsity
        if sp > 0:
            nh = nh + b*nh**(1.+sp)

        # Get estimate and normalize
        if lw:
            w = nw / (sum( nw, axis=0) + eps)
        h = nh / (sum( nh, axis=0) + eps)

    h *= g
    return w.dot(h), w, h

# Separate given trained models
def nmf_sep( M, w1, w2, sp = 0):

    # Fit 'em on mixture
    w = (w1,w2)
    _,_,h = nmf_model( M, w, 300, 0, sp)

    # Get modulator estimates
    q = cumsum( [0, w[0].shape[1], w[1].shape[1]])
    fr = [w[i].dot( h[q[i]:q[i+1],:]) for i in arange( 2)]
    fr0 = hstack(w).dot(h)+eps

    return fr0,fr[0],fr[1]


#
# Learn NN model of a sound using lasagne
#

#import theano
#import theano.tensor as Th
#import downhill

# Training loop
def downhill_train( opt, train, hh, ep, pl):
    import pylab
    from IPython.display import clear_output, display
    from tqdm import tqdm
    cst = []
    st = time.time()
    lt = st
    
    try:
        for tm,_ in tqdm( opt.iterate( train, learning_rate=hh, max_updates=ep, patience=ep, min_improvement=0), total=ep):
            cst.append( tm['loss'])
            if time.time() - lt > 4 and pl is not None:
                nt = time.time()
                epc = len( cst)
                clf()
                pl()
                semilogy( cst), grid( 'on')
                title( 'Cost: %.1e  Speed: %.2f ep/s  Time: %.1f/%.1f' %
                  (cst[-1], epc/(nt-st), nt-st, ep/(epc/(nt-st))) )
                ylabel( 'Cost')
                clear_output( wait=True), show()
                lt = time.time()
    except KeyboardInterrupt:
        pass

    if pl is not None:
        clf()
        nt = time.time()
        epc = len( cst)
        pl()
        semilogy( cst), grid( 'on')
        title( 'Cost: %.1e  Speed: %.2f ep/s  Time: %.1f/%.1f' %
            (cst[-1], epc/(nt-st), nt-st, ep/(epc/(nt-st))) )
        ylabel( 'Cost')
        clear_output( wait=True), show()
    return cst


# Parameterized softplus
def psoftplus( x, p = 1.):
#    return Th.log1p( Th.exp( p*x))/p
    return Th.switch( x < -30./p, 0., Th.switch( x > 30./p, x, Th.log1p( Th.exp( p*x))/p))

#from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, ElemwiseSumLayer
#from lasagne.layers import SliceLayer, get_output, get_all_params, Conv1DLayer
#from lasagne.layers import RecurrentLayer, get_output_shape
#from lasagne.regularization import l1, regularize_layer_params

# Get a Lasagne layer output
def nget( x, s, y):
    if type( s) is list:
        return theano.function( s, squeeze( get_output( x, deterministic=True)), on_unused_input='ignore')( y)
    else:
        return theano.function( [s], squeeze( get_output( x, deterministic=True)), on_unused_input='ignore')( y)



############################################
#########     2-LAYER NETWORK     ##########
############################################

# Learn model
def nn_model( M, K = 20, hh = .0001, ep = 5000, d = 0, wsp = 0.0001, hsp = 0, spb = 3, bt = 0, al='rprop'):

    # Sort out the activation
    from inspect import isfunction
    if isfunction( spb):
        act = spb
    else:
        act = lambda x: psoftplus( x, spb)

    # Copy key variables to GPU
    _M = Th.matrix( '_M')

    # Input and forward transform
    I = InputLayer( shape=(None,M.shape[0]), input_var=_M)

    # First layer is the transform to a non-negative subspace
    H0  = DenseLayer( I, num_units=K, nonlinearity=act, b=None)

    # Optional dropout
    H = DropoutLayer( H0, d)

    # Compute output
    R  = DenseLayer( H, num_units=M.T.shape[1], nonlinearity=act, b=None)

    # Cost function
    Ro = get_output( R)+eps
    cost = Th.mean( _M*(Th.log( _M+eps) - Th.log( Ro)) - _M + Ro)  \
      + wsp*Th.mean( abs( R.W[0])) + hsp*Th.mean( get_output( H0))

    # Train it using Lasagne
    opt = downhill.build( al, loss=cost, inputs=[_M], params=get_all_params( R))
    train = downhill.Dataset( M.T.astype(float64), batch_size=bt)
    er = downhill_train( opt, train, hh, ep, None)

    # Get approximation
    _r = nget( R, _M, M.T.astype( float64)).T
    _h = nget( H, _M, M.T.astype( float64)).T

    return _r,R.W.get_value(),er,_h


# Separate mixture given NN models
def nn_sep( M, W1, W2, hh = .0001, ep = 5000, d = 0, sp =.0001, spb = 3, al='rprop'):

    # Sort out the activation
    from inspect import isfunction
    if isfunction( spb):
        act = spb
    else:
        act = lambda x: psoftplus( x, spb)

    # Get dictionary shapes
    K = [W1.shape[0],W2.shape[0]]

    # GPU cached data
    _M = theano.shared( M.T.astype( float64))
    dum = Th.vector( 'dum')

    # We have weights to discover
    H = theano.shared( sqrt( 2./(K[0]+K[1]+M.shape[1]))*random.rand( M.T.shape[0],K[0]+K[1]).astype( float64))
    fI = InputLayer( shape=(M.T.shape[0],K[0]+K[1]), input_var=H)

    # Split in two pathways
    fW1 = SliceLayer( fI, indices=slice(0,K[0]), axis=1)
    fW2 = SliceLayer( fI, indices=slice(K[0],K[0]+K[1]), axis=1)

    # Dropout?
    dfW1 = DropoutLayer( fW1, dum[0])
    dfW2 = DropoutLayer( fW2, dum[0])

    # Compute source modulators using previously learned dictionaries
    R1  = DenseLayer( dfW1, num_units=M.T.shape[1], W=W1.astype( float64),
      nonlinearity=act, b=None)
    R2  = DenseLayer( dfW2, num_units=M.T.shape[1], W=W2.astype( float64),
      nonlinearity=act, b=None)

    # Add the two approximations
    R = ElemwiseSumLayer( [R1, R2])

    # Cost function
    Ro = get_output( R)+eps
    cost = (_M*(Th.log(_M+eps) - Th.log( Ro+eps)) - _M + Ro).mean() \
       + sp*Th.mean( abs( H)) + 0*Th.mean( dum)

    # Train it using Lasagne
    opt = downhill.build( al, loss=cost, inputs=[dum], params=[H])
    #train = downhill.Dataset( array( [0]).astype(float32), batch_size=0)
    if isinstance( d, list):
        train = downhill.Dataset( array([d[0]]).astype(float64), batch_size=0)
        er = downhill_train( opt, train, hh, ep/2, None)
        train = downhill.Dataset( array([d[1]]).astype(float64), batch_size=0)
        er += downhill_train( opt, train, hh, ep/2, None)
    else:
        train = downhill.Dataset( array([d]).astype(float64), batch_size=0)
        er = downhill_train( opt, train, hh, ep, None)

    # Get outputs
    _r  = nget( R,  dum, array( [0]).astype(float64)).T + eps
    _r1 = nget( R1, dum, array( [0]).astype(float64)).T
    _r2 = nget( R2, dum, array( [0]).astype(float64)).T

    return _r,_r1,_r2,er

# All in one NN separation
def sep_run(data, K, nn = False, sz = 1024, hp = None, s = 0, sw=0, sh=[0,0], dp=0, spb=3, gp=0, arguments=None):
    
                
    # Load sound set
    #random.seed( s)
    #Z = sound_set(3)

    # Front-end details
    #if hp is None:
    #    hp = sz/4
    #wn = reshape( hanning(sz+1)[:-1], (sz,1))**.5

    ## Make feature class
    #FE = sound_feats( sz, hp, wn)
    #al = 'rprop'
    #hh = .0001

    # Learn models
    #M,P = FE.fe( Z[0])
    M = list(data[0])[0][1].numpy().squeeze().transpose()  
    if nn:
        print( 'Using DNN')
        r1,w1,er1,_ = nn_model( M, K[0], hh, 2000, dp, sw, sh[0], spb, al=al)
        print( er1[-1])
    else:
        print( 'Using NMF')
        r1,w1,_ = nmf_model( M, K[0], 100, sp=sw)
    #o = FE.ife( r1, P)
    #c1 = bss_eval( o, 0, array([Z[0]]))
    #print( array( c1))

    M = list(data[1])[0][1].numpy().squeeze().transpose()  
    #M,P = FE.fe( Z[1])
    if nn:
        r2,w2,er2,_ = nn_model( M, K[1], hh, 2000, dp, sw, sh[0], spb, al=al)
        print( er2[-1])
    else:
        r2,w2,_ = nmf_model( M, K[1], 100, sp=sw)
    #o = FE.ife( r2, P)
    #c2 = bss_eval( o, 0, array([Z[1]]))
    #print( array( c2))

    # Separate
    M = list(data[2])[0][0].numpy().squeeze().transpose()  
    P = list(data[2])[0][1].numpy().squeeze().transpose()  

    #M,P = FE.fe( Z[2]+Z[3])
    if nn:
        r,r1,r2,er = nn_sep( M, w1, w2, .0001, 2000, 0, sh[1], spb, al=al)
        print( er[-1])
        #import matplotlib.pyplot as plt
        #plt.hold(True)
        #plt.semilogy( er1, 'b')
        #plt.semilogy( er2, 'g')
        #plt.semilogy( er, 'r')
        #plt.grid('on')
        #plt.legend(['Model1','Model2','Separator'])
    else:
        r,r1,r2 = nmf_sep( M, w1, w2, sw)
    #o1 = FE.ife( r1 * (M/r), P)
    #o2 = FE.ife( r2 * (M/r), P)
    #sxr = bss_eval( o1, 0, vstack( (Z[2],Z[3]))) + bss_eval( o2, 1, vstack( (Z[2],Z[3])))
    #st = [stoi( o1, Z[2], 16000),stoi( o2, Z[3], 16000)]

    temp1_audio, mags1 = ut.mag2spec_and_audio_wiener(r1.transpose(), 
                                                      (r1+r2).transpose(),
                                                      M.transpose(),
                                                      P.transpose(), arguments)
    temp2_audio, mags2 = ut.mag2spec_and_audio_wiener(r2.transpose(), 
                                                      (r1+r2).transpose(),
                                                      M.transpose(),
                                                      P.transpose(), arguments)

    s1 = list(data[2])[0][4].numpy().squeeze()
    s2 = list(data[2])[0][5].numpy().squeeze()
 
    #s1s = (np.split(wavfls1.cpu().numpy(), Nmix, 0))
    #s2s = (np.split(wavfls2.cpu().numpy(), Nmix, 0))

    bss_evals = ut.audio_to_bsseval(temp1_audio, temp2_audio, [s1], [s2])
    bss_df = ut.compile_bssevals(bss_evals) 
    # Return things of note
    # return Z, o1, o2
    #return Z, o1, o2, append( (array(sxr[:3]) + array(sxr[3:]))/2., (st[0]+st[1])/2.)
    return bss_df

############################################
#########      DEEP NETWORK       ##########
############################################

# Learn a deep model
def dnn_model( M, K = [20,20], hh = .0001, ep = 5000, d = 0, wsp = 0.0001, hsp = 0, spb = 3, bt = 0, al='rprop'):

    # Sort out the activation
    from inspect import isfunction
    if isfunction( spb):
        act = spb
    else:
        act = lambda x: psoftplus( x, spb)

    # Copy key variables to GPU
    _M = Th.matrix( '_M')

    # Input and forward transform
    I = InputLayer( shape=(None,M.shape[0]), input_var=_M)

    # Setup the layers
    L = K + [M.T.shape[1]]
    H  = len(L)*[None]
    Hd = len(L)*[None]

    # First layer
    H[0]  = DenseLayer( I, num_units=K[0], nonlinearity=act, b=None)

    # All the rest
    for k in range( 1, len( L)):
        # Optional dropout
        Hd[k-1] = DropoutLayer( H[k-1], d)

        # Next layer
        H[k]  = DenseLayer( Hd[k-1], num_units=L[k], nonlinearity=act, b=None)

    # Cost function
    Ro = get_output( H[-1])+eps
    cost = Th.mean( _M*(Th.log( _M+eps) - Th.log( Ro)) - _M + Ro)
    for k in range( len( L)-1):
        cost += wsp*Th.mean( abs( H[k].W)) + hsp*Th.mean( get_output( H[k]))

    # Train it using Lasagne
    opt = downhill.build( al, loss=cost, inputs=[_M], params=get_all_params( H[-1]))
    train = downhill.Dataset( M.T.astype(float64), batch_size=bt)
    er = downhill_train( opt, train, hh, ep, None)

    # Get approximation
    h = [nget( H[k], _M, M.T.astype( float64)).T for k in range( len( L))]
    w = [H[k].W.get_value() for k in range( len( L))]

    return h,w,er


# Separate mixture given deep models
def dnn_sep( M, W1, W2, hh = .0001, ep = 5000, d = 0, sp =.0001, spb = 3, al='rprop'):

    # GPU cached data
    _M = theano.shared( M.T.astype( float64))
    dum = Th.vector( 'dum')

    # Get layer sizes
    K = []
    for i in range( len( W1)):
        K.append( [W1[i].shape[0],W2[i].shape[0]])
    K.append( [M.T.shape[1],M.T.shape[1]])

    # We have weights to discover, init = 2/(Nin+Nout)
    H = theano.shared( sqrt( 2./(K[0][0]+K[0][1]+M.shape[1]))*random.rand( M.T.shape[0],K[0][0]+K[0][1]).astype( float64))
    fI = InputLayer( shape=(M.T.shape[0],K[0][0]+K[0][1]), input_var=H)

    # Split in two pathways, one for each source's autoencoder
    H1 = (len(W1)+1)*[None]
    H2 = (len(W1)+1)*[None]
    H1[0] = SliceLayer( fI, indices=slice(0,K[0][0]), axis=1)
    H2[0] = SliceLayer( fI, indices=slice(K[0][0],K[0][0]+K[0][1]), axis=1)

    # Put the subsequent layers
    for i in range( len( W1)):
        H1[i+1] = DenseLayer( H1[i], num_units=K[i+1][0], W=W1[i].astype( float64),
        nonlinearity=lambda x: psoftplus( x, spb), b=None)
        H2[i+1] = DenseLayer( H2[i], num_units=K[i+1][1], W=W2[i].astype( float64),
        nonlinearity=lambda x: psoftplus( x, spb), b=None)

    # Add the two approximations
    R = ElemwiseSumLayer( [H1[-1], H2[-1]])

    # Cost function
    Ro = get_output( R)+eps
    cost = Th.mean( _M*(Th.log( _M+eps) - Th.log( Ro)) - _M + Ro) + 0*Th.mean( dum)
    for i in range( len( H1)-1):
        cost += sp*Th.mean( abs( get_output( H1[i]))) + sp*Th.mean( abs( get_output( H2[i])))

    # Train it using Lasagne
    opt = downhill.build( al, loss=cost, inputs=[dum], params=[H])
    train = downhill.Dataset( array([d]).astype(float64), batch_size=0)
    er = downhill_train( opt, train, hh, ep, None)

    # Get outputs
    _r  = nget( R,  dum, array( [0]).astype(float64)).T + eps
    _r1 = nget( H1[-1], dum, array( [0]).astype(float64)).T
    _r2 = nget( H2[-1], dum, array( [0]).astype(float64)).T

    return _r,_r1,_r2,er



# All in one DNN separation
def dnn_sep_run( K, sz = 1024, hp = None, s = 0, sw=0, sh=[0,0], dp=0, spb=3, gp=0):
    # from paris.signal import bss_eval, stoi

    # Load sound set
    random.seed( s)
    Z = sound_set(4)

    # Front-end details
    if hp is None:
        hp = sz/4
    wn = reshape( hanning(sz+1)[:-1], (sz,1))**.5

    # Make feature class
    FE = sound_feats( sz, hp, wn)
    al = 'rprop'
    hh = .0001

    def learn_mod( z):
        M,P = FE.fe( z)
        h,w,e = dnn_model( M, K, .0001, 5000, d=dp, wsp=sw, hsp=sh[0])
        return array( bss_eval( FE.ife( h[-1], P), 0, array([Z[i]]))),e,w

    # Learn models
    w = 2*[None]
    e = 2*[None]
    for i in range(2):
        bs,e[i],w[i] = learn_mod( Z[i])
        print( bs)

    # Separate
    M,P = FE.fe( Z[2]+Z[3])
    r,r1,r2,er = dnn_sep( M, w[0][1:], w[1][1:], .00001, 10000, d=0, sp=sh[1], al='rmsprop')

    #hold(True); semilogy( e[0], 'b'); semilogy( e[1], 'g'); semilogy( er, 'r')
    #grid('on'); legend(['Model1','Model2','Separator'])

    o1 = FE.ife( r1 * (M/r), P)
    o2 = FE.ife( r2 * (M/r), P)
    sxr = bss_eval( o1, 0, vstack( (Z[2],Z[3]))) + bss_eval( o2, 1, vstack( (Z[2],Z[3])))
    st = [stoi( o1, Z[2], 16000),stoi( o2, Z[3], 16000)]
    
    # Return things of note
    # return Z, o1, o2
    return Z, o1, o2, append( (array(sxr[:3]) + array(sxr[3:]))/2., (st[0]+st[1])/2.)






############################################
#########       RNN NETWORK       ##########
############################################

# Learn an RNN model
def rnn_model( M, K = 20, hh = .0001, ep = 5000, d = 0, wsp = 0.0001, hsp = 0, spb = 3, bt = 0, al='rmsprop', t=5):
    # Copy key variables to GPU
    _M = Th.matrix( '_M')

    # Input and forward transform
    I = InputLayer( shape=(None,M.shape[0]), input_var=_M)

    # First layer is the transform to a non-negative subspace
    H0  = DenseLayer( I, num_units=K, nonlinearity=lambda x: psoftplus( x, spb), b=None)

    # Optional dropout
    H = DropoutLayer( H0, d)

    # Compute output
    R  = RecurrentLayer( H, num_units=M.T.shape[1], nonlinearity=lambda x: psoftplus( x, spb), gradient_steps=t, b=None)

    # Cost function
    Ro = get_output( R)+eps
    cost = Th.mean( _M*(Th.log( _M+eps) - Th.log( Ro)) - _M + Ro)  \
      + hsp*Th.mean( get_output( H0))

    # Train it using Lasagne
    opt = downhill.build( al, loss=cost, inputs=[_M], params=get_all_params( R))
    train = downhill.Dataset( M.T.astype(float32), batch_size=bt)
    er = downhill_train( opt, train, hh, ep, None)

    # Get approximation
    _r = nget( R, _M, M.T.astype( float32)).T
    _h = nget( H, _M, M.T.astype( float32)).T

    return _r,(R.W_in_to_hid.get_value(),R.W_hid_to_hid.get_value()),er,_h

# Separate a mixture given RNN models
def rnn_sep( M, W1, W2, hh = .0001, ep = 5000, d = 0, sp =.0001, spb = 3, al='rmsprop', t=5):
    # Get dictionary shapes
    K = [W1[0].shape[0],W2[0].shape[0]]

    # GPU cached data
    _M = theano.shared( M.T.astype( float32))
    dum = Th.vector( 'dum')

    # We have weights to discover
    H = theano.shared( sqrt( 2./(K[0]+K[1]+M.shape[1]))*random.rand( M.T.shape[0],K[0]+K[1]).astype( float32))
    fI = InputLayer( shape=(M.T.shape[0],K[0]+K[1]), input_var=H)

    # Split in two pathways
    fW1 = SliceLayer( fI, indices=slice(0,K[0]), axis=1)
    fW2 = SliceLayer( fI, indices=slice(K[0],K[0]+K[1]), axis=1)

    # Dropout?
    dfW1 = DropoutLayer( fW1, dum[0])
    dfW2 = DropoutLayer( fW2, dum[0])

    # Compute source modulators using previously learned dictionaries
    R1  = RecurrentLayer( dfW1, num_units=M.T.shape[1], b=None,
      W_in_to_hid=W1[0].astype( float32), W_hid_to_hid=W1[1].astype( float32),
      nonlinearity=lambda x: psoftplus( x, spb), gradient_steps=5)
    R2  = RecurrentLayer( dfW2, num_units=M.T.shape[1], b=None,
      W_in_to_hid=W2[0].astype( float32), W_hid_to_hid=W2[1].astype( float32),
      nonlinearity=lambda x: psoftplus( x, spb), gradient_steps=5)

    # Add the two approximations
    R = ElemwiseSumLayer( [R1, R2])

    # Cost function
    Ro = get_output( R)+eps
    cost = (_M*(Th.log(_M+eps) - Th.log( Ro+eps)) - _M + Ro).mean() \
       + sp*Th.mean( abs( H)) + 0*Th.mean( dum)

    # Train it using Lasagne
    opt = downhill.build( al, loss=cost, inputs=[dum], params=[H])
    train = downhill.Dataset( array([d]).astype(float32), batch_size=0)
    er = downhill_train( opt, train, hh, ep, None)

    # Get outputs
    _r  = nget( R,  dum, array( [0]).astype(float32)).T + eps
    _r1 = nget( R1, dum, array( [0]).astype(float32)).T
    _r2 = nget( R2, dum, array( [0]).astype(float32)).T

    return _r,_r1,_r2,er

#
# Separate mixture given convolutive NN models
#

# Learn model using a Lasagne network
def cnn_model( M, K=20, T=1, hh=.0001, ep=5000, d=0, hsp=0.0001, wsp=0, spb=3, bt=0, al='rprop'):
    # Facilitate reasonable convolutions core
    theano.config.dnn.conv.algo_fwd = 'fft_tiling'
    theano.config.dnn.conv.algo_bwd_filter = 'none'
    theano.config.dnn.conv.algo_bwd_data = 'none'

    # Reformat input data
    M3 = reshape( M.astype( float32), (1,M.shape[0],M.shape[1]))

    # Copy key variables to GPU
    _M = Th.tensor3( '_M')

    # Input and forward transform
    I = InputLayer( shape=M3.shape, input_var=_M)

    # First layer is the transform to a non-negative subspace
    H = Conv1DLayer( I, filter_size=T, num_filters=K, pad='same',
        nonlinearity=lambda x: psoftplus( x, spb), b=None)

    # Upper layer is the synthesizer
    R = Conv1DLayer( H, filter_size=T, num_filters=M.shape[0], pad='same',
        nonlinearity=lambda x: psoftplus( x, spb), b=None)

    # Cost function
    Ro = get_output( R)+eps
    cost = Th.mean( _M*(Th.log( _M+eps) - Th.log( Ro)) - _M + Ro) \
      + hsp*Th.mean( get_output( H))

    # Train it using Lasagne
    opt = downhill.build( al, loss=cost, inputs=[_M], params=get_all_params( R))
    train = downhill.Dataset( M3, batch_size=bt)
    er = downhill_train( opt, train, hh, ep, None)

    # Get approximation and hidden state
    _r = squeeze( nget( R, _M, M3))
    _h = squeeze( nget( H, _M, M3))

    return _r,R.W.get_value(),er,_h


# Learn model using a Theano network
def cnn_model_th( M, K=20, T=1, hh=.0001, ep=5000, d=0, wsp=0.0001, dp=0):

    rng = theano.tensor.shared_randomstreams.RandomStreams(0)

    # Shared variables to use
    x = Th.matrix('x')
    y = theano.shared( M.astype( theano.config.floatX))
    d = theano.shared( float32( dp))

    # Network weights
    W0 = theano.shared( sqrt( 2./(K+M.shape[0]))*random.randn( K, M.shape[0]).astype( theano.config.floatX))
    W1 = theano.shared( sqrt( 2./(K+M.shape[0]))*random.randn( M.shape[0], K).astype( theano.config.floatX))

    # First layer is the transform to a non-negative subspace
    h = psoftplus( W0.dot( x), 3.)

    # Dropout
    if dp > 0:
        h *= (1. / (1. - d) * (rng.uniform(size=h.shape) > d).astype( theano.config.floatX)).astype( theano.config.floatX)

    # Second layer reconstructs the input
    l1 = W1.dot( h)
    r = psoftplus( l1, 3.)

    # Approximate input using KL-like distance
    cost = Th.mean( y * (Th.log( y+eps) - Th.log( r+eps)) - y + r) + wsp*Th.mean( abs( W1))

    # Make an optimizer and define the training input
    opt = downhill.build( 'rprop', loss=cost, inputs=[x], params=[W0,W1])
    train = downhill.Dataset( M.astype( theano.config.floatX), batch_size=0)

    # Train it
    er = downhill_train( opt, train, hh, ep, None)

    # Get approximation
    d = 0
    _h,_r = theano.function( inputs = [x], outputs = [h,r], updates = [])( M.astype( theano.config.floatX))
    o = FE.ife( _r, P)
    sxr = bss_eval( o, 0, array([z]))

    return _r,W1.get_value(),_h.get_value(),er


# Lasagne separate
def cnn_sep( M, W1, W2, hh=.0001, ep=5000, d=0, sp=.0001, spb=3, al='rprop'):
    # Facilitate reasonable convolutions core
    theano.config.dnn.conv.algo_fwd = 'fft_tiling'
    theano.config.dnn.conv.algo_bwd_filter = 'none'
    theano.config.dnn.conv.algo_bwd_data = 'none'

    # Reformat input data
    M3 = reshape( M.astype( float32), (1,M.shape[0],M.shape[1]))

    # Copy key variables to GPU
    _M = theano.shared( M3.astype( float32))

    # Get dictionary shapes
    K = [W1.shape[1],W2.shape[1]]
    T = W1.shape[2]

    # We have weights to discover
    H = theano.shared( sqrt( 2./(K[0]+K[1]+M.shape[1]))*random.rand( 1,K[0]+K[1],M.T.shape[0]).astype( float32))
    fI = InputLayer( shape=(1,K[0]+K[1],M.T.shape[0]), input_var=H)

    # Split in two pathways
    H1 = SliceLayer( fI, indices=slice(0,K[0]), axis=1)
    H2 = SliceLayer( fI, indices=slice(K[0],K[0]+K[1]), axis=1)

    # Compute source modulators using previously learned convolutional dictionaries
    R1 = Conv1DLayer( H1, filter_size=T, W=W1, num_filters=M.shape[0], pad='same',
        nonlinearity=lambda x: psoftplus( x, spb), b=None)
    R2 = Conv1DLayer( H2, filter_size=T, W=W2, num_filters=M.shape[0], pad='same',
        nonlinearity=lambda x: psoftplus( x, spb), b=None)

    # Add the two approximations
    R = ElemwiseSumLayer( [R1, R2])

    # Cost function
    dum = Th.vector( 'dum')
    Ro = get_output( R)+eps
    cost = Th.mean(_M*(Th.log(_M+eps) - Th.log( Ro)) - _M + Ro) + 0*Th.mean( dum) + sp*Th.mean( abs( H))

    # Train it using Lasagne
    opt = downhill.build( al, loss=cost, inputs=[dum], params=[H])
    train = downhill.Dataset( array( [0]).astype(float32), batch_size=0)
    er = downhill_train( opt, train, hh, ep, None)

    # Get outputs
    _r  = squeeze( nget( R,  dum, array( [0]).astype(float32))) + eps
    _r1 = squeeze( nget( R1, dum, array( [0]).astype(float32)))
    _r2 = squeeze( nget( R2, dum, array( [0]).astype(float32)))

    return _r,_r1,_r2,er



# SDR, SIR, SAR estimation
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



def drawnow():
    clear_output( wait=True), show()

def imagesc( x, cm = None):
    if cm is None : 
        if amin( x) >= 0 or amax( x) <= 0 : 
            imshow( x, aspect='auto', origin='lower', interpolation='nearest', cmap='bone_r')
        else :
            imshow( x, aspect='auto', origin='lower', interpolation='nearest', cmap='Blues')
            clim( amax( fabs(x)), -amax( fabs(x)) )
    else : 
        imshow( x, aspect='auto', origin='lower', interpolation='nearest', cmap=cm)
    colorbar()
    
def stoi(x, y, fs_signal) :
    import scipy
    from scipy.signal import resample

    # return 1/3 octave band matrix
    def thirdoct( fs, N_fft, numBands, mn) : 
        f   = linspace( 0, fs, N_fft+1)
        f   = f[:N_fft/2+1]
        k   = arange( float( numBands))
        cf  = 2**(k/3)*mn;
        fl  = sqrt( (2.**(k/3)*mn) * 2**((k-1.)/3)*mn)
        fr  = sqrt( (2.**(k/3)*mn) * 2**((k+1.)/3)*mn)
        A   = zeros( (numBands, len( f)) )

        for i in arange( len(cf)) : 
            b     = argmin( (f-fl[i])**2)
            fl[i] = f[b]
            fl_ii = b

            b     = argmin( (f-fr[i])**2)
            fr[i] = f[b]
            fr_ii = b
            A[i,arange(fl_ii,fr_ii)] = 1

        rnk      = sum( A, axis=1)
        numBands = where( (rnk[1:] >= rnk[:-1]) & (rnk[1:] != 0))[-1][-1]+1
        A        = A[:numBands+1,:];
        cf       = cf[:numBands+1];
        return A, cf

    # STFT
    def stdft(x, N, K, N_fft) :
        frames = arange( 0, len(x)-N, K);
        x_stdft = zeros( (len(frames), N_fft) )+1j;
        w = hanning(N);
        for i in arange( len(frames)) :
            ii = arange( frames[i], frames[i]+N);
            x_stdft[i,:] = scipy.fftpack.fft( x[ii]*w, N_fft);
        return x_stdft

    # Remove Silent Frames
    def removeSilentFrames( x, y, rang, N, K) :

        frames = arange( 0, len(x)-N, K)
        w = hanning(N+3)[1:-2]
        msk = 0.*frames

        for j in arange( len( frames)) :
            jj = arange( frames[j], frames[j]+N)
            msk[j] = 20.*log10( linalg.norm( x[jj]*w)/sqrt(N))

        msk = (msk-max(msk)+rang)>0;
        count = 0;
        x_sil = 0.*x;
        y_sil = 0.*y;
        for j in arange( len(frames)) :
            if msk[j] :
                jj_i = arange( frames[j], frames[j]+N)
                jj_o = arange( frames[count], frames[count]+N)
                x_sil[jj_o] = x_sil[jj_o] + x[jj_i]*w
                y_sil[jj_o] = y_sil[jj_o] + y[jj_i]*w
                count += 1

        x_sil = x_sil[:jj_o[-1]]
        y_sil = y_sil[:jj_o[-1]]
        return x_sil, y_sil

    # Correlation function
    def taa_corr( x, y):
        xn = x - mean(x)
        xn = xn / sqrt( sum( xn**2))
        yn = y - mean(y)
        yn = yn / sqrt( sum( yn**2))
        r  = sum( xn * yn)
        return r


    # Force same size
    x = x[:min( len(x), len(y))]
    y = y[:min( len(x), len(y))]
#    if len(x) != len(y):
#        throw('x and y should have the same length');
#        return NaN

    # initialization
    fs          = 16000.;                           # sample rate of proposed intelligibility measure
    N_frame     = 256;                              # window support
    K           = 512;                              # FFT size
    J           = 15;                               # Number of 1/3 octave bands
    mn          = 150;                              # Center frequency of first 1/3 octave band in Hz.
    H,_         = thirdoct(fs, K, J, mn);           # Get 1/3 octave band matrix
    N           = 30;                               # Number of frames for intermediate intelligibility measure (Length analysis window)
    Beta        = -15;                              # lower SDR-bound
    dyn_range   = 40;                               # speech dynamic range

    # resample signals if other samplerate is used than fs
    if fs_signal != fs :
        x = resample( x, float(fs_signal) / fs, 'sinc_best');
        y = resample( y, float(fs_signal) / fs, 'sinc_best');

    # remove silent frames
    x,y = removeSilentFrames( x, y, dyn_range, N_frame, N_frame/2);

    # apply 1/3 octave band TF-decomposition
    x_hat = stdft( x, N_frame, N_frame/2, K);   # apply short-time DFT to clean speech
    y_hat = stdft( y, N_frame, N_frame/2, K);   # apply short-time DFT to processed speech

    x_hat       = x_hat[:, 0:(K/2)+1].T;        # take clean single-sided spectrum
    y_hat       = y_hat[:, 0:(K/2)+1].T;        # take processed single-sided spectrum

    X           = zeros( (J, x_hat.shape[1]) ); # init memory for clean speech 1/3 octave band TF-representation 
    Y           = zeros( (J, y_hat.shape[1]) ); # init memory for processed speech 1/3 octave band TF-representation 

    for i in arange( x_hat.shape[1]) :
        X[:,i] = sqrt( H.dot( abs( x_hat[:,i]))**2);  # apply 1/3 octave bands as described in Eq.(1) [1]
        Y[:,i] = sqrt( H.dot( abs( y_hat[:,i]))**2);

    # loop al segments of length N and obtain intermediate intelligibility measure for all TF-regions
    d_interm    = zeros( (J, X.shape[1]-N) );   # init memory for intermediate intelligibility measure
    c           = 10**(-Beta/20.);              # constant for clipping procedure

    for m in arange( N-1, X.shape[1]) :
        X_seg = X[:, arange( m-N, m)];          # region with length N of clean TF-units for all j
        Y_seg = Y[:, arange( m-N, m)];          # region with length N of processed TF-units for all j
        alpha = sqrt( sum( X_seg**2, axis=1) / sum(Y_seg**2, axis=1)); # obtain scale factor for normalizing processed TF-region for all j
        alpha = array( [alpha]).T
        aY_seg = Y_seg * alpha; # obtain \alpha*Y_j(n) from Eq.(2) [1]
        for j in arange( J) :
            Y_prime          = minimum( aY_seg[j,:], X_seg[j,:]+X_seg[j,:]*c);  # apply clipping from Eq.(3)
            d_interm[j,m-N]  = taa_corr( X_seg[j,:], Y_prime);         # obtain correlation coeffecient from Eq.(4) [1]

    d = mean( d_interm);  # combine all intermediate intelligibility measures as in Eq.(4) [1]
    return d



def nn_sep_ae( m, w1, w2, hh = .001, ep = 5000, sp =.1, dp = 0.0, spb = 3, al='rprop'):
    from numpy import random
    import theano
    # from matplotlib.pyplot import gcf, clf, semilogy, grid, title, show
    from deep_sep_expr3 import downhill_train
    rng = theano.tensor.shared_randomstreams.RandomStreams(0)

    # Dropout parameters
    d = theano.shared( float32( dp))

    # Plot to make while training
    def pl():
        clf()
        gcf().set_size_inches(6,2)
        semilogy( cst); grid( 'on'); title( 'Cost: %f, Epoch: %d' % (cst[-1], len( cst)))
        drawnow()

        

    # Sort out the activation
    from inspect import isfunction
    if isfunction( spb):
        act = spb
    else:
        act = lambda x: psoftplus( x, spb)
        
    w_cat = hstack((w1,w2));
    K = [w1.shape[1], w2.shape[1]];
    # W2m = theano.shared(w_cat.astype(float64))
    
    W1m = theano.shared(random.rand( w_cat.shape[1], w_cat.shape[0]).astype(float64))
#     W1z = theano.shared((linalg.pinv(w_cat)).astype(float64))
    
    M = theano.tensor.matrix('M');
    Hm = psoftplus(W1m.dot( M),spb);
    # Dropout
    if dp > 0:
        Hm *= (1. / (1. - d) * (rng.uniform(size=Hm.shape) > d).astype( theano.config.floatX)).astype( theano.config.floatX)

    W2s1 = theano.shared(hstack((w_cat[:,0:K[0]],zeros(w2.shape))).astype(float64));
    W2s2 = theano.shared(hstack((zeros(w1.shape),w_cat[:,K[0]:K[0]+K[1]])).astype(float64));
    
    M1 = psoftplus(W2s1.dot(Hm),spb);
    M2 = psoftplus(W2s2.dot(Hm),spb);
    M_out = M1 + M2;
    
    # -------------or----------------
    
    # M_out = psoftplus((W2s1 + W2s2).dot( Hm),spb);
    # M2 = psoftplus(M_out - psoftplus(W2s1.dot(Hm),spb),spb);
    # M1 = psoftplus(M_out - psoftplus(W2s2.dot(Hm),spb),spb);
    
    cost = theano.tensor.mean( M_out * (theano.tensor.log( M_out+eps) - theano.tensor.log( M+eps)) - M_out + M) \
           + 0.01*theano.tensor.mean( abs( W1m)**1) + 0.01*theano.tensor.mean( abs( Hm)**1) 
    #cost = theano.tensor.mean( M * (theano.tensor.log( M+eps) - theano.tensor.log( M_out+eps)) - M + M_out) \
    #       + 0.1*theano.tensor.mean( abs( Hm)**1)    + 1*theano.tensor.mean( abs( W2m)**2) 
    opt = downhill.build(al, loss = cost, params = [W1m], inputs = [M]); # params = W1m
    train = downhill.Dataset(m.astype(float64), batch_size = m.shape[0]); # batch_size = m.shape[0]
    cst = [];
    lt = time.time()
    for tm,_ in opt.iterate( train, learning_rate = hh, max_updates=ep, patience = ep):
        cst.append( tm['loss'])
        if time.time() - lt > 2:
            pl()
            lt = time.time()
        
    pl()
    # W2s1 = theano.shared(hstack((w1,zeros(w2.shape))).astype(float64));
    # W2s2 = theano.shared(hstack((zeros(w1.shape),w2)).astype(float64));
    
    #W2s1 = theano.shared(hstack((W2m.eval()[:,0:K[0]],zeros(w2.shape))).astype(float64));
    #W2s2 = theano.shared(hstack((zeros(w1.shape),W2m.eval()[:,K[0]:K[0]+K[1]])).astype(float64));
    #M1 = psoftplus(W2s1.dot(Hm),spb);
    #M2 = psoftplus(W2s2.dot(Hm),spb);
    
    nn_nmf_sep  = theano.function(inputs = [M], outputs = [Hm, M1, M2, M_out], updates = []);
    h1m, m1, m2, m_out = nn_nmf_sep(m.astype( float64));

    subplot( 2, 1, 1); imagesc( m1**.4); title( 'Source 1');
    subplot( 2, 1, 2); imagesc( m2**.4); title( 'Source 2');
    # subplot( 2, 2, 3); plot( h1z[0:Kx].T); title( 'Latent representation for Source 1');
    # subplot( 2, 2, 4); plot( h1z[Kx:Kx+Ky].T); title( 'Latent representation for Source 2');
    tight_layout()

    return m_out, m1, m2, cst


def nn_model_ae( x, Kx, learning_rate = .001, ep = 5000, dp = 0.0, spb = 3, al='rprop'):
    # Train NSAE for Source 1
    # Define NMF network
    
    rng = theano.tensor.shared_randomstreams.RandomStreams(0)


    # Latent dimensions
    
    def pl():
        clf()
        gcf().set_size_inches(6,2)
        semilogy( cst); grid( 'on'); title( 'Cost: %f, Epoch: %d' % (cst[-1], len( cst)))
        drawnow()


    # Dropout parameters
    d = theano.shared( float64( dp))

    # I/O container
    X = theano.tensor.matrix('X')

    # Weight matrices
    W1x = theano.shared( random.rand( Kx, x.shape[0]).astype( float64))
    W2x = theano.shared( random.rand( x.shape[0], Kx).astype( float64))

    # Get latent variables
    Hx = psoftplus( W1x.dot( X), spb)
    # Hx = act( W1x.dot( X))
    
    # Dropout
    if dp > 0:
        Hx *= (1. / (1. - d) * (rng.uniform(size=Hx.shape) > d).astype( theano.config.floatX)).astype( theano.config.floatX)

    # Get approximation
    Zx = psoftplus( W2x.dot( Hx), spb)
    # Zx = act( W2x.dot( Hx))

    # Low rank reconstruction should match smoothed amplitudes, use sparse W1
    cost = theano.tensor.mean( X * (theano.tensor.log( X+eps) - theano.tensor.log( Zx+eps)) - X + Zx) \
           + 1*theano.tensor.mean( abs( W2x)**2) +0.01*theano.tensor.mean( abs( Hx))

    # Make an optimizer and define the inputs
    opt = downhill.build( al, loss=cost, params = [W1x, W2x], inputs=[X])
    train = downhill.Dataset( x.astype( float64), batch_size = x.shape[0])


    # Train and show me the progress
    cst = []
    lt = time.time()
    for tm, _ in opt.iterate( train, learning_rate=.001, max_updates=ep, patience = ep):
        cst.append( tm['loss'])
        if time.time() - lt > 4:
            pl()
            lt = time.time()
    pl()

    # Show me
    nn_nmf = theano.function( inputs=[X], outputs=[Zx,Hx,W2x], updates = [])
    z,h,w = nn_nmf( x.astype( float64))

    subplot( 2, 1, 1); imagesc( x**.4); title( 'Input 1');
    subplot( 2, 1, 2); imagesc( z**.4); title( 'Approximation');
    subplot( 2, 2, 3); plot( W2x.get_value()); title( 'NN bases');
    subplot( 2, 2, 4); plot( h.T); title( 'Latent representation');
    tight_layout()
    return w,z
