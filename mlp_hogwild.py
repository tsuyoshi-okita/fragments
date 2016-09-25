#
##  THEANO_FLAGS=floatX=float32 python mlp_hogwild.py
#

import gzip
import os
import sys
import timeit
import numpy
import numpy as np
import gzip
import pickle
import time
import codecs
from collections import OrderedDict
from multiprocessing import Process, Manager

#from sklearn.datasets import fetch_mldata
#from sklearn.cross_validation import train_test_split
#import matplotlib.pyplot as plt

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

def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def numpy_floatX(data):
    return numpy.asarray(data, dtype='float32')    

def grouping(listA, groupNum):
    length=len(listA)
    groupNumTmp = length / groupNum
    groupNumMod = length % groupNum
    a=[listA[i:i+groupNumTmp+1] for i in range(0, groupNumMod * (groupNumTmp+1), groupNumTmp+1)]
    b=[listA[i:i+groupNumTmp] for i in range(groupNumMod * (groupNumTmp+1), length, groupNumTmp)]
    return a+b

def _p(pp, name):
    return '%s_%s' % (pp, name)

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


def prepare_data(seqs, labels, maxlen=None):
    return seqs, labels

def load_data():
    f=gzip.open('mnist.pkl.gz','r')    
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    train_set_x0, train_set_y0 = train_set
    valid_set_x0, valid_set_y0 = valid_set
    test_set_x0, test_set_y0 = test_set
    rval = [(train_set_x0,train_set_y0), (valid_set_x0, valid_set_y0), (test_set_x0, test_set_y0)]
    return rval

layers = {'ff': ('self.param_init_fflayer', 'self.fflayer'),
          'gru': ('self.param_init_gru', 'self.gru_layer'),
          'gru_cond': ('self.param_init_gru_cond', 'self.gru_cond_layer'),
          }

class ff:
    def tanh(self,x):
        import theano.tensor as T            
        return T.tanh(x)
    def linear(self,x):
        return x

    def get_layer(self,name):
        fns = layers[name]
        return (eval(fns[0]), eval(fns[1]))
    
    def init_tparams(self,params):
        import theano
        from theano.tensor.shared_randomstreams import RandomStreams                
        tparams = OrderedDict()
        for kk, pp in params.iteritems():
            tparams[kk] = theano.shared(params[kk], name=kk)
        print tparams
        return tparams

    def itemlist(self,tparams):
        return [vv for kk, vv in tparams.iteritems()]

    def param_init_fflayer(self,options, params, prefix='ff', nin=None, nout=None, ortho=True):
        if nin is None:
            nin = options['dim_proj']
        if nout is None:
            nout = options['dim_proj']
        params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho).astype('float32')
        params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')
        return params

    def fflayer(self,tparams, state_below, options, prefix='rconv', activ='lambda x: T.tanh(x)', **kwafrgs):
        import theano.tensor as T                
        return eval(activ)(T.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')])

    def pred_error(self, f_pred, data, iterator, verbose=False):
        valid_err = 0
        for _, valid_index in iterator:
            x = [data[0][v] for v in valid_index]
            y = [data[1][v] for v in valid_index]
            preds = f_pred(x).argmax(axis=1)
            targets = numpy.array(data[1])[valid_index]
            try:
                valid_err += (preds == targets).sum()
            except:
                valid_err += (preds == targets)
        valid_err = 1. - numpy_floatX(valid_err) / len(data[0])
        return valid_err

    def init_params(self,options):
        params = OrderedDict()
        params = self.get_layer('ff')[0](options, params, prefix='ff_mlp1',nin=options['nin'], nout=options['n_hidden'])
        params = self.get_layer('ff')[0](options, params, prefix='ff_mlp2',nin=options['n_hidden'], nout=options['n_hidden'])
        params = self.get_layer('ff')[0](options, params, prefix='ff_mlp3',nin=options['n_hidden'], nout=options['nout'])
        return params

    def build_model(self,tparams, options):
        from theano.tensor.shared_randomstreams import RandomStreams
        import theano
        import theano.tensor as T
        import theano.tensor as tensor
        
        opt_ret = dict()
        trng = RandomStreams(1234)
        use_noise = theano.shared(numpy_floatX(0.))

        x = T.matrix('x', dtype='float32')
        y = T.ivector('y')
        n_samples=x.shape[0]
        n_dim =x.shape[1]
        mlp1 = self.get_layer('ff')[1](tparams, x, options, prefix='ff_mlp1', activ='self.tanh')
        mlp2 = self.get_layer('ff')[1](tparams, mlp1, options, prefix='ff_mlp2', activ='self.tanh')
        mlp3 = self.get_layer('ff')[1](tparams, mlp2, options, prefix='ff_mlp3', activ='self.tanh')

        showDim('mlp3',mlp3)     #mlp3.ndim=2  (False, False) TensorType(float32, matrix)
        probs = T.nnet.softmax(mlp3)

        off = 1e-8
        if probs.dtype == 'float16':
            off = 1e-6

        cost = -tensor.log(probs[tensor.arange(n_samples), y] + off).mean()

        f_pred_prob = theano.function([x], probs, name='f_pred_prob', allow_input_downcast=True)
        f_pred = theano.function([x], probs.argmax(axis=1), name='f_pred', allow_input_downcast=True, on_unused_input='ignore')    
    
        return use_noise, x, y, cost, f_pred, f_pred_prob

    def sgd(self,lr, tparams, grads, inp, cost):
        import theano
        import theano.tensor as T        
        
        gshared = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tparams.iteritems()]
	gsup = [(gs, g) for gs, g in zip(gshared, grads)]
        f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile, allow_input_downcast=True)
        pup = [(p, (p - lr * g)) for p, g in zip(self.itemlist(tparams), gshared)]
        f_update = theano.function([lr], [], updates=pup, profile=profile)
        return f_grad_shared, f_update
    #    f_update = theano.function([lr], [], updates=pup, profile=profile, mode='DebugMode')

    def adagrad(self, lr, tparams, grads, inp, cost):
        import theano
        import theano.tensor as T        
        
        zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),name='%s_grad' % k) for k, p in tparams.iteritems()]
        running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),name='%s_rgrad2' % k) for k, p in tparams.iteritems()]
        updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir' % k) for k, p in tparams.iteritems()]
        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, rg2 + (g ** 2)) for rg2, g in zip(running_grads2, grads)]
        f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up, profile=profile, allow_input_downcast=True)
        updir_new = [- lr * zg / (T.sqrt(rg2 + 1e-6)) for zg, rg2 in zip(zipped_grads, running_grads2)]
        rg2up = [(rg2, rg2 + (ud ** 2)) for rg2, ud in zip(running_grads2, updir_new)]
        param_up = [(p, p + ud) for p, ud in zip(self.itemlist(tparams), updir_new)]    
        f_update = theano.function([lr], [], updates=rg2up+param_up, profile=profile)
        return f_grad_shared, f_update
    #    f_update = theano.function([lr], [], updates=updir_new, on_unused_input='ignore', profile=profile)

    def adam(self, lr, tparams, grads, inp, cost):
        import theano
        import theano.tensor as T        
        
        gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) for k, p in tparams.iteritems()]
        gsup = [(gs, g) for gs, g in zip(gshared, grads)]
        f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile, allow_input_downcast=True)
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

    def adadelta(self, lr, tparams, grads, inp, cost):
        import theano
        import theano.tensor as T        
        
        zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad' % k) for k, p in tparams.iteritems()]
        running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2' % k) for k, p in tparams.iteritems()]
        running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2' % k) for k, p in tparams.iteritems()]
        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]
        f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up, profile=profile, allow_input_downcast=True)
        updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(self.itemlist(tparams), updir)]
        f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore', profile=profile)
        return f_grad_shared, f_update

    def rmsprop(self, lr, tparams, grads, inp, cost):
        import theano
        import theano.tensor as T        
        
        zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),name='%s_grad' % k) for k, p in tparams.iteritems()]
        running_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad' % k) for k, p in tparams.iteritems()]
        running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2' % k) for k, p in tparams.iteritems()]
        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]
        f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up, profile=profile, allow_input_downcast=True)
        updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir' % k) for k, p in tparams.iteritems()]
        updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
        param_up = [(p, p + udn[1]) for p, udn in zip(self.itemlist(tparams), updir_new)]
        f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=profile)
        return f_grad_shared, f_update
    
    def __init__(self,shared_args,private_args):
        import theano
        import theano.tensor as T
        from theano import config, pp
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        import theano.printing
        from theano.tensor.shared_randomstreams import RandomStreams
        import theano.sandbox.cuda
        
        theano.sandbox.cuda.use(private_args['gpu'], force=True)
        model_options = shared_args[0]
        total, accuracies, batchsize = [[],[],20]
        
        patience=model_options['patience']
        max_epochs=model_options['max_epochs']
        dispFreq=model_options['dispFreq']
        decay_c=model_options['decay_c']
        lrate=model_options['lrate']
        optimizer=model_options['optimizer']
        batch_size=model_options['batch_size']
        valid_batch_size=model_options['valid_batch_size']
        saveto=model_options['saveto']
        validFreq=model_options['validFreq']
        saveFreq=model_options['saveFreq']
        sampleFreq=model_options['sampleFreq']
        lrate=model_options['lrate']
        model_options['n_epochs']=1000
        n_epochs=model_options['n_epochs']
        max_epochs=model_options['max_epochs']        
        
        counter000=0
#        model_options = shared_args[0]
        print (model_options)    
        total, accuracies, batchsize = [[],[],20]
        datasets = load_data()
#        batch_size=1024
        train_set_x0, train_set_y0 = datasets[0]        
        valid_set_x0, valid_set_y0 = datasets[1]
        test_set_x0, test_set_y0 = datasets[2]    
        n_train_batches = train_set_x0.shape[0] // batch_size
#        train_set_x00 = grouping(train_set_x0, model_options['threadNum'])[private_args['threadID']]
#        train_set_y00 = grouping(train_set_y0, model_options['threadNum'])[private_args['threadID']]    
        
        params = self.init_params(model_options)
        tparams=self.init_tparams(params)
        use_noise, x, y, cost, f_pred, f_pred_prob = self.build_model(tparams, model_options)    
        inps = [x, y]
        print 'Building f_log_probs...',
        index = T.scalar(name='index', dtype='int64')    
        #    f_log_probs = theano.function(inps, cost, profile=profile)
        cost=cost.mean()
        
        print 'Computing gradient...',
        grads = T.grad(cost, wrt=self.itemlist(tparams))
        print 'Done'
        
        lr = T.scalar(name='lr')
        f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)            

        epoch = 0
        accuracies = []
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
        patience = 5000 
        patience_increase = 2
        improvement_threshold = 0.995
        history_errs=[]

        for eidx in xrange(max_epochs):
            n_samples=0
            kf = get_minibatches_idx(len(train_set_x0), batch_size, shuffle=True)
            for ii, train_index in kf:
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
#                m=f_pred_prob(x0)
#                m2=f_pred(x0)            
                cost = f_grad_shared(x0, y0)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'bad cost detected: ', cost
                    return 1., 1., 1.
                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, private_args['gpu']
                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = self.pred_error(f_pred_prob, [train_set_x0, train_set_y0], kf)
                    valid_err = self.pred_error(f_pred_prob, [valid_set_x0,valid_set_y0], kf_valid)
                    test_err =  self.pred_error(f_pred_prob, [test_set_x0,test_set_y0], kf_test)
                    print('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err, private_args['gpu'])

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
        train_err = self.pred_error(f_pred_prob, [train_set_x0, train_set_y0], kf)
        valid_err = self.pred_error(f_pred_prob, [valid_set_x0,valid_set_y0], kf_valid)
        test_err =  self.pred_error(f_pred_prob, [test_set_x0,test_set_y0], kf_test)

        print(('Optimization complete with best validation score of %f %%,''with test performance %f %%')% (best_validation_loss * 100., test_score * 100.))
        print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time)))
        
def main(job_id, params):
    manager = Manager()
    args = manager.list()
    args.append({})
    shared_args = args[0]
    p_args,options,p=[{},{},{}]

    shared_args['optimizer']=params['optimizer'][0]
    shared_args['nin']=params['nin'][0]
    shared_args['nout']=params['nout'][0]
    shared_args['n_hidden']=params['n_hidden'][0]
    shared_args['num_threads']=params['num_threads'][0]            
    shared_args['lrate']=0.000001
    shared_args['dataset']='mnist.pkl.gz'
    shared_args['n_epochs']=1000
    shared_args['max_epochs']=1000    
    shared_args['lrate']=0.000001
    shared_args['validFreq']=100
    shared_args['dispFreq']=10
    shared_args['saveFreq']=100
    shared_args['sampleFreq']=100
    shared_args['patience']=1000
    shared_args['batch_size']=100
    shared_args['valid_batch_size']=100        
    shared_args['decay_c']=0.
    shared_args['saveto']='model_mlp.pkl'
    num_threads=shared_args['num_threads']
    shared_args['threadNum']=num_threads
    
    for x in range(num_threads):
        p_args[x] = {}
    for x in range(num_threads):
        p_args[x]['offset'] = num_threads
        p_args[x]['threadID'] = x
        if (x < num_threads/2):
            p_args[x]['gpu'] = 'gpu1'
        else:
            p_args[x]['gpu'] = 'gpu0'
    epoch_time = time.time()

    args[0] = shared_args
    p=[Process(target=ff, args=(args,p_args[x],)) for x in range(num_threads)]
    [p[x].start() for x in range(num_threads)]
    [p[x].join() for x in range(num_threads)]
    epoch_time = time.time() - epoch_time
    print ("Final model:", epoch_time, "seconds")
    
if __name__ == '__main__':
    main(0, {'nin': [784],
             'nout': [10],
             'n_hidden': [500],
             'optimizer': ['self.adadelta'],
#             'optimizer': ['self.adam'],             
#             'optimizer': ['self.sgd'],                                       
             'num_threads': [2],            
#             'num_threads': [4],            597 
#             'num_threads': [16],           723
             'alg': [1]})

