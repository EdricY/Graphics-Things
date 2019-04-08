import os
import time
import inspect

def show_warning(text):
    print('>>> WARNING <<< : %s' % text)

def put_your_code_here(fn):
    if not hasattr(put_your_code_here, 'reported'):
        put_your_code_here.reported = set()
    frame = inspect.currentframe().f_back
    filename = os.path.basename(frame.f_code.co_filename)
    linenum = frame.f_lineno
    fnname = fn.__name__ #frame.f_code.co_name
    text = '%s (%s:%d)' % (fnname, filename, linenum)
    if text not in put_your_code_here.reported:
        put_your_code_here.reported.add(text)
        show_warning('Function %s not (fully) implemented' % text)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

def timed_call(label):
    def wrapper(fn):
        def wrapped(*args, **kwargs):
            time_beg = time.time()
            ret = fn(*args, **kwargs)
            time_end = time.time()
            time_delta = time_end - time_beg
            print('Timing: %0.2fs, %s' % (time_delta, label))
            return ret
        return wrapped
    return wrapper