import numpy as np

from .tomograms import Tomograms

class Likelihood():
    """
    frame_model (F_dri):
        basic:      W_ri
        fluence:    w_d W_ri
        background: w_d W_ri + B_di

    likelihood (logR_dr):
        Poisson:      sum_i K_di log(F_dri) - F_dri
        fluence_free: sum_i K_di log(F_dri) - K_d log(F_dr)

    where K_d  = sum_i K_di
          F_dr = sum_i F_dri

    No background:
        likelihood = 'fluence_free'
        frame_model = 'basic'
            logR_dr = \sum_i K_di logW_ri - K_d log(sum_i W_ri)

        likelihood = 'Poisson'
        frame_model = 'basic'
            logR_dr = \sum_i K_di logW_ri - sum_i W_ri

        likelihood = 'Poisson'
        frame_model = 'fluence'
            logR_dr = \sum_i K_di logW_ri - w_d sum_i W_ri

    Background:
        likelihood = 'Poisson'
        frame_model = 'background'
            logR_dr = \sum_i K_di logF_dri - F_dr

        likelihood = 'fluence_free'
        frame_model = 'background'
            logR_dr = \sum_i K_di logF_dri - K_d logF_dr

    Computation strategy:
        sparse_K
        gpu
    """
    def __init__(
            self,
            tomo,
            K_di,
            frame_model='basic',
            likelihood='fluence_free',
            sparse_K=False,
            gpu=False,
            **kwargs
            ):
        self.tomo = tomo
        self.K_di = K_di

        self.frame_model = frame_model
        self.likelihood = likelihood

    def calculate(self):
        """
        """
        # make tomograms
        W_ri = []
        for r in range(self.tomo.shape[0]):
            W_ri.append(self.tomo.calculate_tomogram(r))

        W_ri = np.array(W_ri)

        self.wsums_r = np.sum(W_ri, axis=1)

        # calculate dot product
        logR_dr = np.dot(self.K_di[:], np.log(W_ri).T)

        self.offset(logR_dr)

        return logR_dr

    def offset(self, logR_dr, drange=None, rrange=None):
        if drange is None:
            drange = [0, self.K_di.shape[0]]

        if rrange is None:
            rrange = [0, self.tomo.shape[0]]

        d0, d1 = drange
        r0, r1 = rrange

        print(f'******************************************************hello*************************************************')
        print(f'{self.frame_model=}')
        print(f'{self.likelihood=}')

        # offset
        if (
                self.frame_model == 'basic'
                and self.likelihood == 'fluence_free'
                ):
            print(f'{logR_dr.max()=}')
            logR_dr -= self.K_di.data_sum[d0:d1, None] \
                    * np.log(self.wsums_r[r0:r1])[None, :]
            print(f'{logR_dr.max()=}')

        elif (
                self.frame_model == 'basic'
                and self.likelihood == 'Poisson'
                ):
            logR_dr -= self.wsums_r[r0:r1]

        elif (
                self.frame_model == 'fluence'
                and self.likelihood == 'Poisson'
                ):
            logR_dr -= self.tomo.fluence[d0:d1, None] \
                    * self.wsums_r[None, r0:r1]

        else:
            raise ValueError(f'{self.frame_model=} and {self.likelihood=} are not supported')


def calculate_logR(config):

    """
    basic all in memory cpu process
    """
    for c in config['classes']:
        if not c['update_logR']:
            continue

        c['mapper'].load_coords(c['P_data'].mask)

        tomos = Tomograms(
                c['mapper'],
                c['model'],
                c['P_data'].C_i,
                c['fluence'])

        L = Likelihood(tomos, c['P_data'], **c)

        c['logR_dr'] = L.calculate()

