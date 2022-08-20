from textwrap import wrap
import numpy as np
import time
import torch


def bench(enable, count=100):

    def wrapper(func):

        def inner_wrapper(*args, **kwargs):
            if enable:
                time_list = []
                for i in range(count):
                    torch.cuda.synchronize()
                    begin = time.time()
                    func(*args, **kwargs)
                    torch.cuda.synchronize()
                    end = time.time()
                    time_list.append(end - begin)

                print("AVG [ms]:", np.mean(time_list[10:]) * 1000)
            else:
                return func(*args, **kwargs)

        return inner_wrapper

    return wrapper