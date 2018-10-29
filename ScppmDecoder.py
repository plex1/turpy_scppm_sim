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

    def decode(self, y, n_data, expected_data=[]):
        """
        Turbo decoding of an scppm frame

        Parameters
        ----------
        y: list containing 1s and 0s
            data stream as a list of bits

        n_data: integer
            length of data

        expected_data: list containing 1s and 0s
            expected data for bit error rate calcualtion


        Returns
        -------
        u_hard: list containing 1s and 0s | hard decisions of decoded data

        errors_iter: ist with length iterations containig floats between [0,1]
            errors per iteration
        """

        # initialize variables
        Lext = [0] * self.il.get_length()
        ext_scale = 0.75
        errors_iter = [0] * self.iterations

        for i in range(self.iterations):

            # first half iteration --------------------------------------------------------------

            # inner
            a, cout = self.inner_siso.decode(Lext, y, self.il.get_length())
            a_ext = [ext_scale * (x - y) for x, y in zip(a, Lext)]

            # de-interleaver
            x = self.il.deinterleave(a_ext)

            # second half iteration --------------------------------------------------------------

            # outer
            u, cout = self.outer_siso.decode([0] * n_data, x, n_data)
            c_ext = [ext_scale * (x - y) for x, y in zip(cout, x)]

            # interleaver
            Lext = self.il.interleave(c_ext)

            # hard output
            u_hard = [int(x > 0) for x in u]

            if len(expected_data) > 0:  # ber calculation

                error_rate = sum([abs(x - y) for x, y in zip(expected_data, u_hard)]) / n_data
                errors_iter[i] = error_rate

                if error_rate == 0:  # stopping criteria
                    return (u_hard, errors_iter)

                print("iter: " + str(i) + " error rate: " + str(error_rate))

        return (u_hard, errors_iter)
