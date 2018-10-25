#! /usr/bin/env python
# title           : ScppmDecoder.py
# description     : This class implements a SCPPM decoder. Its input is an interleaver and two siso decoder instances.
# author          : Felix Arnold
# python_version  : 3.5.2


class ScppmDecoder(object):

    def __init__(self, interleaver, inner_siso, outer_siso):

        self.inner_siso = inner_siso
        self.outer_siso = outer_siso
        self.il = interleaver
        self.iterations = 6

    def decode(self, y, n_data, n_i, len_x, data_u=[]):

        # initialize variables
        Lext = [0] * n_i
        ext_scale = 0.7
        errors_iter = [0] * self.iterations

        for i in range(self.iterations):

            # first half iteration --------------------------------------------------------------

            # inner
            a, cout = self.inner_siso.decode(Lext, y, n_i)
            a_ext = [x - y for x, y in zip(a, Lext)]

            # de-interleaver
            x = self.il.deinterleave(a_ext)

            # second half iteration --------------------------------------------------------------

            # outer
            u, cout = self.outer_siso.decode([0] * n_data, x[0:len_x], n_data)

            # interleaver
            Lext = self.il.interleave(cout + [-10] * (n_i - len_x))
            Lext = [ext_scale * x for x in Lext]

            # hard output
            u_hard = [int(x > 0) for x in u]

            if len(data_u) > 0:  # ber calculation

                error_rate = sum([abs(x - y) for x, y in zip(data_u, u_hard)]) / n_data
                errors_iter[i] = error_rate

                if error_rate == 0:  # stopping criteria
                    return (u_hard, errors_iter)

                print("iter: " + str(i) + " error rate: " + str(error_rate))

        return (u_hard, errors_iter)
