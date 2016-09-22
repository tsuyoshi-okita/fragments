#!/usr/local/bin python
#! -*- coding: utf-8 -*-

import gzip
import os
import sys
import timeit
import time
import pickle
from collections import OrderedDict
import codecs

import numpy
import numpy as np

import theano
import theano.tensor as T
import theano.tensor as tensor
from theano import config, pp
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.compile.debugmode import DebugMode
import theano.printing

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

def _p(pp, name):
    return '%s_%s' % (pp, name)

def tanh(x):
    return T.tanh(x)

def linear(x):
    return x

def l2norm(X):
    norm = T.sqrt(T.pow(X, 2).sum(1))
    X /= norm[:, None]
    return X

def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

layers = {'ff': ('param_init_fflayer', 'fflayer'),}

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

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

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    print tparams
    return tparams

def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None, ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho).astype('float32')
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')
    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: T.tanh(x)', **kwafrgs):
    return eval(activ)(T.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')])

def pred_error(f_pred, data, iterator, verbose=False):
    valid_err = 0
    for _, valid_index in iterator:
        x = [data[0][v] for v in valid_index]
        preds = f_pred(x).argmax(axis=1)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])
    return valid_err
    
def init_params(options):
    params = OrderedDict()
    params = get_layer('ff')[0](options, params, prefix='ff_mlp1',nin=options['nin'], nout=options['n_hidden'])
    params = get_layer('ff')[0](options, params, prefix='ff_mlp2',nin=options['n_hidden'], nout=options['n_hidden'])
    params = get_layer('ff')[0](options, params, prefix='ff_mlp3',nin=options['n_hidden'], nout=options['nout'])
    return params

def build_model(tparams, options):
    opt_ret = dict()
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy_floatX(0.))

    x = T.matrix('x', dtype='float32')
    y = T.ivector('y')

    n_samples=x.shape[0]
    n_dim =x.shape[1]
    
    mlp1 = get_layer('ff')[1](tparams, x, options, prefix='ff_mlp1', activ='tanh')
    mlp2 = get_layer('ff')[1](tparams, mlp1, options, prefix='ff_mlp2', activ='tanh')
    mlp3 = get_layer('ff')[1](tparams, mlp2, options, prefix='ff_mlp3', activ='tanh')

    showDim('mlp3',mlp3)     #mlp3.ndim=2  (False, False) TensorType(float32, matrix)
    probs = T.nnet.softmax(mlp3)

    off = 1e-8
    if probs.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(probs[tensor.arange(n_samples), y] + off).mean()

    f_pred_prob = theano.function([x], probs, name='f_pred_prob',allow_input_downcast=True)
    f_pred = theano.function([x], probs.argmax(axis=1), name='f_pred',allow_input_downcast=True, on_unused_input='ignore')    
    
    return use_noise, x, y, cost, f_pred, f_pred_prob

def sgd(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile, allow_input_downcast=True)
    pup = [(p, (p - lr * g)) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)
    return f_grad_shared, f_update


def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile,allow_input_downcast=True)
    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8
    updates = []
    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (T.sqrt(fix2) / fix1)
    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore', profile=profile)
    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad' % k) for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2' % k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2' % k) for k, p in tparams.iteritems()]
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]
    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up, profile=profile, allow_input_downcast=True)
    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]
    f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore', profile=profile)
    return f_grad_shared, f_update



def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),name='%s_grad' % k) for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad' % k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2' % k) for k, p in tparams.iteritems()]
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]
    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up, profile=profile, allow_input_downcast=True,on_unused_input='ignore')
    updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir' % k) for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=profile)
    return f_grad_shared, f_update

def adagrad(lr, tparams, grads, inp, cost):
    # stores the current grads
    gshared = [theano.shared(np.zeros_like(p.get_value(), dtype=theano.config.floatX), name='%s_grad' % k) for k, p in tparams.iteritems()]
    grads_updates = zip(gshared, grads)
    # stores the sum of all previous grads squared
    hist_gshared = [theano.shared(np.zeros_like(p.get_value(), dtype=theano.config.floatX), name='%s_grad' % k) for k, p in tparams.iteritems()]
    rgrads_updates = [(rg, rg + T.sqr(g)) for rg, g in zip(hist_gshared, grads)]
    # calculate cost and store grads
    f_grad_shared = theano.function(inp, cost, updates=grads_updates + rgrads_updates, allow_input_downcast=True,on_unused_input='ignore')
    # apply actual update with the initial learning rate lr
    fudge = 1e-6  # to achieve numerical stability
    updates = [(p, p - (lr/(T.sqrt(rg) + fudge))*g) for p, g, rg in zip(tparams.values(), gshared, hist_gshared)]
    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore')
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
    
def train(nin=512, nout=10, n_hidden=500, lrate=0.000001, n_epochs=1000,
          decay_c=0.1,
          max_epochs=1000,
          dispFreq=1000,
          batch_size=200,          
          valid_batch_size=64,          
          validFreq=100,
          saveFreq=1000,
          patience=100,
          optimizer='adadelta',
    ):

    model_options = locals().copy()
    datasets = load_data()

    train_set_x0, train_set_y0 = datasets[0]        
    valid_set_x0, valid_set_y0 = datasets[1]
    test_set_x0, test_set_y0 = datasets[2]    
    n_train_batches = train_set_x0.shape[0] // batch_size
    print n_train_batches
#    print valid_set_x0
    
    params = init_params(model_options)
    tparams=init_tparams(params)
    use_noise, x, y, cost, f_pred, f_pred_prob = build_model(tparams, model_options)
    inps=[x,y]
    
    print 'Building f_log_probs...',
    index = T.scalar(name='index', dtype='int64')
    cost=cost.mean()
    
    print 'Done'
    print 'Computing gradient...',
    grads = T.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    lr = T.scalar(name='lr')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)    

    print 'Optimization start...'
    kf_valid = get_minibatches_idx(len(valid_set_x0), valid_batch_size)
    kf_test = get_minibatches_idx(len(test_set_x0), valid_batch_size)    
    uidx=0
    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size
    best_p = None
    epoch = 0    
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.time()
    history_errs=[]
    bad_counter =0
    estop = False
    outfile = codecs.open('tmptmp', 'w','utf-8')
        
    for eidx in xrange(max_epochs):
        n_samples=0
        kf = get_minibatches_idx(len(train_set_x0), batch_size, shuffle=True)
        for ii, train_index in kf:
            outfile.write(str(ii) +'\n')
            use_noise.set_value(1.)
            uidx += 1
            x0 = [train_set_x0[t] for t in train_index]
            y0 = [train_set_y0[t] for t in train_index]
            x0=numpy.array(x0)
            y0=numpy.array(y0)
            n_samples += x0.shape[1]            
            if x0 is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue
            ud_start = time.time()
            m=f_pred_prob(x0)
            m2=f_pred(x0)            
            cost = f_grad_shared(x0, y0)
            f_update(lrate)

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'bad cost detected: ', cost
                return 1., 1., 1.
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                train_err = pred_error(f_pred_prob, [train_set_x0, train_set_y0], kf)
                valid_err = pred_error(f_pred_prob, [valid_set_x0,valid_set_y0], kf_valid)
                test_err =  pred_error(f_pred_prob, [test_set_x0,test_set_y0], kf_test)
                print('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)

                history_errs.append([valid_err, test_err])
                if (best_p is None or
                    valid_err <= numpy.array(history_errs)[:, 0].min()):
                    best_p = unzip(tparams)
                    bad_counter = 0
                
                if (len(history_errs) > patience and
                    valid_err >= numpy.array(history_errs)[:-patience,0].min()):
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break
#        print 'Seen %d samples' % n_samples
        
        if estop:
            break

    outfile.close()
    end_time = timeit.default_timer()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)
    use_noise.set_value(0.)
    
    kf = get_minibatches_idx(len(train_set_x0), batch_size, shuffle=True)                        
    train_err = pred_error(f_pred_prob, [train_set_x0, train_set_y0], kf)
    valid_err = pred_error(f_pred_prob, [valid_set_x0,valid_set_y0], kf_valid)
    test_err =  pred_error(f_pred_prob, [test_set_x0,test_set_y0], kf_test)

    print(('Optimization complete with best validation score of %f %%,''with test performance %f %%')% (best_validation_loss * 100., test_score * 100.))
    print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time)))
#    print(('The code for file ' +os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

def main(argv, params):
    print params
    validerr = train(nin=params['nin'][0],
                     nout=params['nout'][0],
                     n_hidden=params['n_hidden'][0],
                     lrate=params['learning_rate'][0],
                     optimizer=params['optimizer'][0],
                     batch_size=params['batch_size'][0],
                     valid_batch_size=params['valid_batch_size'][0])
    return validerr

if __name__ == '__main__':
    main(sys.argv, {'nin': [784],
             'nout': [10],
             'n_hidden': [500],
             'learning_rate': [0.001],
#             'optimizer': ['adadelta'],
#             'optimizer': ['adagrad']                    
#             'optimizer': ['sgd']
                    'optimizer': ['adam'],
#             'optimizer': ['rmsprop']
#             'batch_size': [200],
          'batch_size': [784],                              
          'valid_batch_size': [800]
    })


