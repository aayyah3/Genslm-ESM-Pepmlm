import numpy as np

class MaskingScheduler():
    def __init__(self, 
                 schedule_type: str='linear', 
                 max_iters: int = 10,
                 max_length: int = 256,
                 manual_ratio: float = -1,
                 manual_rate: float = -1):
        self.schedule_map = {
            'linear': self.linear_schedule,
            'power': self.power_schedule,
            'cosine': self.cosine_schedule,
            'gaussian': self.gaussian_schedule,
            'exponential': self.exponential_schedule,
            'fixed': self.fixed_schedule
        }
        self.schedule_function = self.schedule_map[schedule_type]
        self.max_iters = max_iters
        self.max_length = max_length
        self.manual_ratio = manual_ratio

    """
    Each schedule should satisfy that the masking rate is bounded
    between 0 and 1, and that if t==max_iters, the masking rate is 0.
    if t==0, the masking rate is 1.
    """
    def linear_schedule(self, t, manual_ratio: float = -1):
        if manual_ratio >= 0:
            return 1 - manual_ratio
        return 1 - t/self.max_iters
    def power_schedule(self, t, manual_ratio: float = -1):
        if manual_ratio >= 0:
            return 1 - manual_ratio**2
        return 1 - (t/self.max_iters)**2
    def cosine_schedule(self, t, manual_ratio: float = -1):
        if manual_ratio >= 0:
            return np.cos(manual_ratio * np.pi / 2.)
        return np.cos(t/self.max_iters * np.pi / 2.)
    def gaussian_schedule(self, t, manual_ratio: float = -1):
        # draw from random distribution with mean 0.3 and std 0.3
        # if manual_ratio is set, use that value instead
        if manual_ratio >= 0:
            return manual_ratio
        return np.clip(np.random.normal(loc=0.3, scale=0.3), 0.01, 0.99)

    def exponential_schedule(self, t, manual_ratio: float = -1):
        # draw from exponential distribution with mean 0.3
        # if manual_ratio is set, use that value instead
        if manual_ratio >= 0:
            return manual_ratio
        return np.random.exponential(scale=0.3)
    def fixed_schedule(self, t, manual_ratio: float = -1):
        # if manual_ratio is set, use that value instead
        if manual_ratio >= 0:
            return manual_ratio
        return self.manual_ratio

    def estimate_tokens_to_recover(self, t, max_length: int):
        return int(self.schedule_function(t) * max_length)
    
    def __call__(self, t, manual_ratio: float = -1):
        rate = self.schedule_function(t, manual_ratio)
        return np.clip(rate, 0, 1)
    
if __name__=='__main__':
    # test various schedulers, ensure returned masking rates are
    # bounded 0 <= rate <= 1
    scheduler = MaskingScheduler(schedule_type='linear', max_iters=10)
    # log mean, min, max, std of masking rates
    masking_rates = []
    for t in range(11):
        rate = scheduler(t)
        masking_rates.append(rate)
        print(f"linear {t} ::: {rate:0.4f}")
    print(f"linear mean: {np.mean(masking_rates):0.4f}")
    print(f"linear min: {np.min(masking_rates):0.4f}")
    print(f"linear max: {np.max(masking_rates):0.4f}")

    scheduler = MaskingScheduler(schedule_type='power', max_iters=10)
    masking_rates = []
    for t in range(11):
        rate = scheduler(t)
        masking_rates.append(rate)
        print(f"power {t}  ::: {rate:0.4f}")
    print(f"power mean: {np.mean(masking_rates):0.4f}")
    print(f"power min: {np.min(masking_rates):0.4f}")
    print(f"power max: {np.max(masking_rates):0.4f}")

    scheduler = MaskingScheduler(schedule_type='cosine', max_iters=10)
    masking_rates = []
    for t in range(11):
        rate = scheduler(t)
        masking_rates.append(rate)
        print(f"cosine {t} ::: {rate:0.4f}")
    print(f"cosine mean: {np.mean(masking_rates):0.4f}")
    print(f"cosine min: {np.min(masking_rates):0.4f}")
    print(f"cosine max: {np.max(masking_rates):0.4f}")

    scheduler = MaskingScheduler(schedule_type='gaussian', max_iters=10)
    masking_rates = []
    for t in range(11):
        rate = scheduler(t)
        masking_rates.append(rate)
        print(f"gaussian {t} ::: {rate:0.4f}")
    print(f"gaussian mean: {np.mean(masking_rates):0.4f}")
    print(f"gaussian min: {np.min(masking_rates):0.4f}")
    print(f"gaussian max: {np.max(masking_rates):0.4f}")

    scheduler = MaskingScheduler(schedule_type='exponential', max_iters=10)
    masking_rates = []
    for t in range(11):
        rate = scheduler(t)
        masking_rates.append(rate)
        print(f"exponential {t} ::: {rate:0.4f}")
    print(f"exponential mean: {np.mean(masking_rates):0.4f}")
    print(f"exponential min: {np.min(masking_rates):0.4f}")
    print(f"exponential max: {np.max(masking_rates):0.4f}")