#! /usr/bin/env python
# title           : ScppmDecoder.py
# description     : This class implements a SCPPM decoder. Its input is an interleaver and two siso decoder instances.
# author          : Felix Arnold
# python_version  : 3.5.2

import numpy as np
from turpy.ConvEncoder import ConvEncoder


class ScppmDecoder(object):

    def __init__(self, interleaver, inner_siso, outer_siso):

        self.inner_siso = inner_siso
        self.outer_siso = outer_siso
        self.il = interleaver
        self.iterations = 6

    def decode(self, y, n_data, n_i, len_x, data_u=[]):

        #n_i = 64
        #n_data = 24
        #len_x = 52
        # initialize variables
        Lext = [0] * n_i
        ext_scale = 0.7


        errors_iter = [0] * self.iterations

        for i in range(self.iterations):

            # first half iteration

            # inner
            a = self.inner_siso.decode(Lext, y, [0] * n_i, n_i)
            a_ext = [1*(x- Lext[index]) for index, x in enumerate(a)]
            # a_ext = a-Lext

            #test
            cve = ConvEncoder(self.outer_siso.trellis)
            ahat=self.il.interleave(cve.encode(data_u)+ [0] * (n_i-len_x))
            a_hard = [int(x > 0) for x in a]
            Lext_hard = [int(x > 0) for x in Lext]
            a_ext_hard = [int(x > 0) for x in a_ext]
            error_rate_hard_a = sum([abs(x - y) for x, y in zip(a_hard, ahat)]) / n_i


            errors_response_a = 0
            responses_a = 0
            responses_Lext = 0
            errors_response_Lext = 0
            responses_a_ext = 0
            errors_response_a_ext = 0
            for k in range(len(ahat)):
                if a[k] != 0:
                    responses_a += 1
                    if ahat[k] != a_hard[k]:
                        errors_response_a +=1
                if Lext[k] != 0:
                    responses_Lext += 1
                    if ahat[k] != Lext_hard[k]:
                        errors_response_Lext +=1
                if a_ext[k] != 0:
                    responses_a_ext += 1
                    if ahat[k] != a_ext_hard[k]:
                        errors_response_a_ext +=1

            # de-interleaver
            x = self.il.deinterleave(a_ext)

            # test
            x_hard = [int(x1 > 0) for x1 in x]
            xhat = cve.encode(data_u) + [0] * (n_i - len_x)
            error_rate_x = sum([abs(x1 - y1) for x1, y1 in zip(x_hard, xhat)]) / n_i

            # second half iteration

            # outer
            u = self.outer_siso.decode([0] * (n_data + self.outer_siso.trellis.K - 1), x[0:len_x], [0] * (n_data + self.outer_siso.trellis.K - 1), n_data)

            # interleaver
            Lext = self.il.interleave(self.outer_siso.lue + [-10] * (n_i-len_x))
            Lext = [ext_scale* Lext[index] for index, x in enumerate(Lext)]

            llr_max = 16
            Lext = [llr_max if x > llr_max else x for x in Lext]
            Lext = [-llr_max if x < -llr_max else x for x in Lext]


            # hard output
            u_hard = [int(x > 0) for x in u]

            if len(data_u) > 0:  # ber calculation
                error_rate = sum([abs(x - y) for x, y in zip(data_u, u_hard)]) / n_data
                errors_iter[i] = error_rate

                # test
                responses_u = 0
                errors_response_u = 0
                for k in range(len(data_u)):
                    if u[k] != 0:
                        responses_u += 1
                        if data_u[k] != u_hard[k]:
                            errors_response_u += 1

                if error_rate == 0:  # stopping criteria
                    return (u_hard, errors_iter)
                print("iter: "+ str(i) +" error rate: " + str(error_rate))

        return (u_hard, errors_iter)
