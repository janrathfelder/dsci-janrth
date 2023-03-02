
class SampleSize(object):
    
    def __init__(self):
        pass
    
    def find_sample_size(self,
                         population_size: int,
                         error_margin: float,
                         confidence_level=.95,
                         standard_deviation=.5,
                    ) -> int:
        '''
        Finds the required sample size based on input given.
        
        The input params are defined as:
            :param population_size: the size (estimate) of the full population
            :param error margin: percentage of error which is acceptable (suggested to be between .03 and .1)
            :param confidence level: confidence level for the margin of error (normally between .9 and .99)
            :param standard deviation: defines the shape of the Normal Distribution (suggested to be .5 when no prior information is available)
        
        '''
        if confidence_level>=.99:
            z_value=2.567
        elif ((confidence_level>=.95) and (confidence_level<.99)):
            z_value=1.96
        elif ((confidence_level>=.9) and (confidence_level<.95)):
            z_value=1.645
        else:
            print('confidence level needs to be between .9 and 1')
        n = ((z_value**2 * standard_deviation*(1-standard_deviation)) / error_margin**2) / (1+(z_value**2 * standard_deviation*(1-standard_deviation))/(error_margin**2 * population_size))
        return round(n)