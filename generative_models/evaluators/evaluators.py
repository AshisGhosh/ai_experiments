class Evaluator:
    def __init__(self, sampler, metric_fns):
        self.sampler = sampler
        self.metric_fns = metric_fns

    def evaluate(self, batch_size, **kwargs):
        samples, _ = self.sampler.sample(batch_size, **kwargs)
        return {name: fn(samples) for name, fn in self.metric_fns.items()}
