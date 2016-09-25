import os
import time
import sys
from collections import OrderedDict
import gzip
import re
import cPickle
import cPickle as pickle
import gzip

import numpy
import numpy as np

import theano
import theano.tensor as tensor
import theano.printing
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

floatX = theano.config.floatX

epsilon = 1e-8
profile=False
SEED = 123
numpy.random.seed(SEED)

def showDim(x1,x):
    if type(x)==list:
        print str(x1)+'.type=list', x
    elif x.ndim==1:
        print str(x1)+'.ndim=1 '+' '+str(x.broadcastable)+' '+str(x.type)
    elif x.ndim==2:
        print str(x1)+'.ndim=2 '+' '+str(x.broadcastable)+' '+str(x.type)
    elif x.ndim==3:
        print str(x1)+'.ndim=3 '+' '+str(x.broadcastable)+' '+str(x.type)
    elif x.ndim==0:
        print str(x1)+'.ndim=0 '+' '+str(x.broadcastable)+' '+str(x.type)
    else:
        print 'something else'+str(x1)

def numpy_floatX(data):
    return numpy.asarray(data, dtype='float32')    

def kld_unit_mvn(mu, var):
    return (mu.shape[1] + tensor.sum(tensor.log(var), axis=1) - tensor.sum(tensor.square(mu), axis=1) - tensor.sum(var, axis=1)) / 2.0

def log_diag_mvn(mu, var, reconstructed_x, x):
#    k = mu.shape[1]
    logpxz = (-1 / 2.0) * np.log(2 * np.pi) - 0.5 * tensor.sum(tensor.log(var)) - (0.5 * (1.0 / var) * (x - reconstructed_x) **2).sum(axis=1)
    return logpxz

def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    #    print n
    idx_list = numpy.arange(n, dtype="int32")
    if shuffle:
        numpy.random.shuffle(idx_list)
    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size
    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])
    return zip(range(len(minibatches)), minibatches)

def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

def linear(x):
    return x

def tanh(x):
    return tensor.tanh(x)

def _p(pp, name):
    return '%s_%s' % (pp, name)
    
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
}

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')
    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')])

def init_params(options):
    params = OrderedDict()
    params = get_layer('ff')[0](options, params, prefix='enc_mlp', nin=options['features'], nout=options['hu_encoder'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='enc_mu', nin=options['hu_encoder'], nout=options['n_latent'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='enc_logvar', nin=options['hu_encoder'], nout=options['n_latent'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='dec_mlp', nin=options['n_latent'], nout=options['hu_decoder'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='dec_mu', nin=options['hu_decoder'], nout=options['features'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='dec_logvar', nin=options['hu_decoder'], nout=options['features'], ortho=False)
#    params = get_layer('ff')[0](options, params, prefix='dec_be', nin=options['hdim'], nout=options['nout'], ortho=False)
    return params

def build_model(tparams, options, x_train):
    opt_ret = dict()
    print tparams
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))
    
    x = tensor.matrix('x', dtype=floatX)
    y = tensor.vector('y', dtype='int32')                
    epoch = tensor.iscalar("epoch")
    batch_size = x.shape[0]
    n_samples=x.shape[0]
    n_dim = x.shape[1]

    showDim('x',x)
    # Gaussian
    def encoder(x, tparams, options):
        enc_mlp = get_layer('ff')[1](tparams, x, options, prefix='enc_mlp', activ='tanh')
        mu = get_layer('ff')[1](tparams, enc_mlp, options, prefix='enc_mu', activ='linear')
        log_sigma = get_layer('ff')[1](tparams, enc_mlp, options, prefix='enc_logvar', activ='linear')
        return mu, log_sigma
    mu, log_sigma = encoder(x, tparams, options)
    def sampler(mu, log_sigma):
        seed = 42
        if "gpu" in theano.config.device:
            srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
        else:
            srng = tensor.shared_randomstreams.RandomStreams(seed=seed)
        eps = srng.normal(mu.shape)
        z = mu + tensor.exp(0.5 * log_sigma) * eps
        return z
    z = sampler(mu, log_sigma)
    def decoder(x, z, tparams, gaussian, options):
        dec_mlp = get_layer('ff')[1](tparams, z, options, prefix='dec_mlp', activ='tanh')
#        if options['gaussian']:
        if gaussian:
            reconstructed_x = get_layer('ff')[1](tparams, dec_mlp, options, prefix='dec_mu', activ='linear')
            log_sigma_decoder = get_layer('ff')[1](tparams, dec_mlp, options, prefix='dec_logvar', activ='linear')
            var=tensor.exp(log_sigma_decoder)
            logpxz = (-(0.5 * np.log(2 * np.pi) + 0.5 * log_sigma_decoder) -
                      0.5 * ((x - reconstructed_x)**2 / tensor.exp(log_sigma_decoder))).sum(axis=1, keepdims=True)
#            logpxz = log_diag_mvn(mu, var, reconstructed_x, x)
        else:
            dec_mlp = get_layer('ff')[1](tparams, dec_mlp, options, prefix='dec_mu', activ='linear')
            reconstructed_x = tensor.nnet.sigmoid(dec_mlp)
            logpxz = - tensor.nnet.binary_crossentropy(reconstructed_x, x).sum(axis=1, keepdims=True)
        return reconstructed_x, logpxz
    reconstructed_x, logpxz = decoder(x,z, tparams, options['gaussian'], options)

    KLD = kld_unit_mvn(mu, numpy.exp(log_sigma))
#    logpx = tensor.mean(logpxz + KLD)
    logpx = -tensor.mean(logpxz + KLD)
    kld_mean= -KLD.mean()

#    f_likelihood = theano.function([x], logpx)
    f_likelihood = theano.function([x], [logpx,kld_mean])
    f_encode = theano.function([x], z)
    f_decode = theano.function([z], reconstructed_x)

    return logpx, kld_mean, x, z, epoch, f_likelihood, f_encode, f_decode, use_noise

def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile, on_unused_input='ignore')

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore', profile=profile)
    return f_grad_shared, f_update

def load_data():
    f=gzip.open('mnist.pkl.gz','r')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    train_set_x0, train_set_y0 = train_set
    valid_set_x0, valid_set_y0 = valid_set
    test_set_x0, test_set_y0 = test_set
    rval = [(train_set_x0,train_set_y0), (valid_set_x0, valid_set_y0), (test_set_x0, test_set_y0)]
    return rval

def train(hu_encoder = 400,
          hu_decoder = 400,
          n_latent = 20,
          b1=0.05,
          b2=0.001,
          batch_size=100,
          valid_batch_size=100,          
          lrate=0.001,
          n_epochs = 40,                
          lam=0,
          dispFreq=10,
          validFreq=1000,
          saveFreq=1000,          
          max_epochs=10000,
          sigma_init = 0.01,
          gaussian = True,
          optimizer='sgd',
          path = "./"):

    model_options = locals().copy()
    datasets = load_data()
    x_train, t_train = datasets[0]        
    x_valid, t_valid_set_y0 = datasets[1]
    x_test, t_test = datasets[2]    
    n_train_batches = x_train.shape[0] // batch_size
    [N, features] = x_train.shape
    model_options['features']=features
    model_options['N']=N    

    params = init_params(model_options)
    tparams = init_tparams(params)
    options=model_options

    print "Build model",
    cost_logpx, kld_mean, x, z, epoch0, f_likelihood, f_encode, f_decode, use_noise = build_model(tparams, model_options, x_train)
    print "Done"

    gradients = tensor.grad(cost_logpx, tparams.values(), disconnected_inputs='ignore')

    inps2=[x]    
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps2, [cost_logpx, kld_mean])    
    print 'Done'

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, gradients, inps2, cost_logpx)    
    print 'Done'
    
    epoch = 0
    LB_list = []

    if os.path.isfile(path + "params.pkl"):
        p_list = cPickle.load(open(path + "/params.pkl", "rb"))
        for name in p_list.keys():
            tparams[name].set_value(p_list[name].astype('float32'))
        LB_list = np.load(path + "LB_list.npy")
        epoch = len(LB_list)
    print "Optimization"
    kf_valid = get_minibatches_idx(len(x_valid), valid_batch_size)
    kf_test = get_minibatches_idx(len(x_test), valid_batch_size)    
    uidx=0
    if validFreq == -1:
        validFreq = len(x_train) / batch_size
    if saveFreq == -1:
        saveFreq = len(x_train) / batch_size
        
    for eidx in xrange(max_epochs):        
        kf = get_minibatches_idx(len(x_train), batch_size, shuffle=True)
#        epoch += 1        
        for ii, train_index in kf:
            use_noise.set_value(1.)            
            start = time.time()
            uidx += 1
            x0 = [x_train[t] for t in train_index]
            y0 = [t_train[t] for t in train_index]
            x0=numpy.array(x0)
            y0=numpy.array(y0)

            logpx0, kld_mean0 = f_likelihood(x0)            
            cost = f_grad_shared(x0)
            f_update(lrate)
            if numpy.mod(uidx, dispFreq) == 0:
                print "Epoch {0} iter: {1}, cost: {2}, ut: {3}, kld: {4}, logpx0: {5}".format(epoch, uidx, cost, time.time() - start, kld_mean0, logpx0)
            if numpy.mod(uidx, saveFreq) == 0:
#                pp=[re.sub('vaelayer_','',name) for name,p in params.items()]
                pp=[name for name,p in params.items()]
                cPickle.dump({name: tparams[name].get_value() for name in pp}, open(path + "/params.pkl", "wb"))
        
        valid_LB = f_likelihood(x_valid)
        print "LowerBound on validation set: {0}".format(valid_LB)
        return

def main(argv, params):
    print params
    validerr = train(hu_encoder=params['hu_encoder'][0],
                     hu_decoder=params['hu_decoder'][0],
                     n_latent=params['n_latent'][0],
                     lrate=params['learning_rate'][0],
                     optimizer=params['optimizer'][0],
                     gaussian=params['gaussian'][0],
                     batch_size=params['batch_size'][0])
    return validerr

if __name__ == '__main__':
    main(sys.argv, {'hu_encoder': [400],                     
                    'hu_decoder': [400],                     
             'n_latent': [20],
             'learning_rate': [0.001],
                    'gaussian': [False],
#             'optimizer': ['adadelta'],
#             'optimizer': ['adagrad']                    
#                    'optimizer': ['sgd'],
                    'optimizer': ['adam'],
#             'optimizer': ['rmsprop']
#             'batch_size': [200],
                    'batch_size': [100]                              
    })
