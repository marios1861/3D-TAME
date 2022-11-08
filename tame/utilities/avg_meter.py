from queue import Queue


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, use_ema: bool = True, a: float = 0.001, k: int = 10):
        self.use_ema = use_ema
        if not use_ema:
            self.values: Queue[float] = Queue(maxsize=k)
        self.a = a
        self.k = k
        self.reset()

    def reset(self):
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
        self.init = True

    def update(self, val: float):
        self.val = val
        if self.init:
            self.avg = self.val
            self.init = False
        else:
            if self.use_ema:
                self.avg = self.a * self.val + (1 - self.a) * self.avg
            else:
                if not self.values.full():
                    self.values.put_nowait(self.val)
                    self.avg = self.val
                else:
                    last_val = self.values.get_nowait()
                    self.values.put_nowait(self.val)
                    self.avg = self.avg + (self.val - last_val) / self.k

    def __call__(self) -> float:
        return self.avg

    def __str__(self) -> str:
        return str(self.avg)
