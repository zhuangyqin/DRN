"""
extened data io
"""
from multiprocessing import Process, Queue
from multiprocessing.dummy import Process as Thread, Queue as ThreadQueue
import mxnet as mx
from mxnet.io import DataDesc, DataBatch
from utils.tictoc import tic,toc
import logging
import numpy as np

class PrefetchergroupIter(mx.io.DataIter):
    def __init__(self, dataiter, rename_data=None, rename_label=None,num_threads=4,queue_size=8,fetcher_type="thread"):
        super(PrefetchergroupIter, self).__init__()
        self.dataiter = dataiter
        self.iters = [dataiter]
        self.rename_data = rename_data
        self.rename_label = rename_label
        self.batch_size = len(self.provide_data) * self.provide_data[0][0][1][0]
        self.num_threads = num_threads
        self.queue_size = queue_size
        self.size = dataiter.size

        # multi threads to reading the  data and support the buffer reading
        creators = {'process': [Queue, BatchFetcherProcess],
                    'thread': [ThreadQueue, BatchFetcherThread],}[fetcher_type]
        self.queue = creators[0](queue_size)
        self.fetcher_thread_creator = creators[1]

        self.current_batch = None
        self.procs = []
        self.running = False
        self.reset()

    def reset(self):

        self.cursor = -self.batch_size
        # if has, close the threading
        if self.procs != []:
            for proc in self.procs:
                proc.join()
        # reset the data
        self.dataiter.reset()

        num_threads = self.num_threads
        num_batches = self.dataiter.size // self.batch_size

        batch_splits = [1. * (num_batches - _) / num_threads for _ in xrange(num_threads)]
        batch_splits = np.ceil(batch_splits).astype(np.int32)
        logging.info("batch split into {}".format(batch_splits))

        for tid in xrange(num_threads):
            left = batch_splits[:tid].sum()
            # which batch would be load
            thread_perm = xrange(left,left + batch_splits[tid])
            fetcher = BatchFetcher(self.queue,
                                   self.dataiter,
                                   thread_perm)
            proc = self.fetcher_thread_creator(fetcher)
            proc.daemon = True
            proc.start()
            self.procs.append(proc)
        self.running = True


    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        if self.rename_data is None:
            return sum([i.provide_data for i in self.iters], [])
        else:
            return sum([[
                DataDesc(r[x.name], x.shape, x.dtype)
                if isinstance(x, DataDesc) else DataDesc(*x)
                for x in i.provide_data
            ] for r, i in zip(self.rename_data, self.iters)], [])


    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        if self.rename_label is None:
            return sum([i.provide_label for i in self.iters], [])
        else:
            return sum([[
                DataDesc(r[x.name], x.shape, x.dtype)
                if isinstance(x, DataDesc) else DataDesc(*x)
                for x in i.provide_label
            ] for r, i in zip(self.rename_label, self.iters)], [])


    def iter_next(self):
        ## when return the None stop iteration
        self.cursor += self.batch_size
        # so the cursor reset must be -batch_size
        self.running = (self.cursor + self.batch_size <= self.size)
        return self.running

    def next(self):
        tic()
        if self.iter_next():
            self.current_batch = self.queue.get()
            logging.debug("loading the data time in Group Inter: {}".format(toc()))
            return self.current_batch
        else:
            raise StopIteration

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad


class BatchFetcher(object):
    def __init__(self, queue, dataiter, perm):
        self._queue = queue
        self._dataiter = dataiter
        self._perm = perm
        self._perm_len = len(perm)
        self._cursor = -1
        
    def iter_next(self):
        self._cursor += 1
        return self._cursor + 1 <= self._perm_len
    
    def run(self):
        while self.iter_next():
            batch_ind = self._perm[self._cursor]
            # put it to the queue
            self._queue.put(self._dataiter[batch_ind])


class BatchFetcherProcess(Process):
    def __init__(self, fetcher):
        super(BatchFetcherProcess, self).__init__()
        self._fetcher = fetcher

    def run(self):
        self._fetcher.run()
        
class BatchFetcherThread(Thread):
    def __init__(self, fetcher):
        super(BatchFetcherThread, self).__init__()
        self._fetcher = fetcher

    def run(self):
        self._fetcher.run()

