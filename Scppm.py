#! /usr/bin/env python
# title           : Scppm.py
# description     : This file implements scppm encoders and decoders
# author          : Felix Arnold
# python_version  : 3.5.2

import numpy as np
from itertools import islice
from turpy.ConvEncoder import ConvEncoder
from turpy.Trellis import Trellis


class PpmModulator(object):

    def __init__(self, m, b=0):
        self.m = m
        self.b = b
        self.m_b = int(np.log2(self.m))

    def modulate_symbol(self, symbol_num):
        enc = [0] * (self.m + self.b)
        enc[symbol_num] = 1
        return enc

    def modulate_stream(self, data):
        return list(flatten(self.modulate_symbol(bin2dec(x)) for x in grouped(data, self.m_b)))

    def demodulate(self, symbol):
        return symbol.index(max(symbol[0:self.m]))

    def demodulate_stream(self, e):
        return list(flatten(dec2bin(self.demodulate(x), self.m_b) for x in grouped(e, self.m + self.b)))


class ScppmEncoder(object):

    def __init__(self, trellis, interleaver, modulator):
        self.state = 0
        self.trellis = trellis
        self.interleaver = interleaver
        self.modulator = modulator
        self.r = 3

    def accu(self, d):
        trellis_accu = Trellis([[1, 0]], [0, 1])
        cve = ConvEncoder(trellis_accu)
        return cve.encode(d, False)

    def encode(self, d):
        self.cve = ConvEncoder(self.trellis)
        d = self.cve.encode(d)  # encoding incl zero termination
        len_x = len(d)
        d = d + [0] * (len(self.interleaver.perm) - len(d))
        d = self.interleaver.interleave(d)  # interleaving
        N_i = len(d)
        d = self.accu(d)
        e = self.modulator.modulate_stream(d)  # modulation
        return [e, len_x, N_i]


import numpy as np


class PpmTrellis(object):

    def __init__(self, modulator):
        self.Ns = 2  # number of states
        self.Nb = 2 * modulator.m  # number of branches
        self.r = modulator.m
        self.rsc = 0
        self.wb = int(np.log2(modulator.m))
        self.mod = modulator
        self.get_dat_precalc = []
        self.get_enc_bits_precalc = []
        self.precalc()

    def get_rate(self):
        return self.r

    def get_next_state(self, branch):
        return int((branch & 2 ** (self.wb - 1)) != 0)

    def get_prev_state(self, branch):
        return int((branch & 2 ** (self.wb)) != 0)

    def get_prev_branches(self, state):
        return np.array([x + state * 2 ** (self.wb - 1) for x in range(2 ** (self.wb - 1))] + [
            x + state * 2 ** (self.wb - 1) + 2 ** self.wb for x in range(2 ** (self.wb - 1))])

    def get_next_branches(self, state):
        return np.array([x + state * 2 ** (self.wb) for x in range(2 ** (self.wb))])

    def get_next_branch(self, state, dat):
        print("not implemented")

    def get_enc_bits(self, branch):
        return self.mod.modulate_symbol(branch & (2 ** self.wb - 1))

    def get_dat(self, branch):
        return self._dec2bin(
            (branch & (2 ** self.wb - 1)) ^ (((branch & (2 ** (self.wb - 1) - 1)) << 1) + self.get_prev_state(branch)),
            self.wb)

    def _dec2bin(self, val, k):
        bin_val = []
        for j in range(k):
            bin_val.append(val & 1)
            val = val >> 1
        return bin_val

    def precalc(self):
        self.get_dat_pc = [self.get_dat(x) for x in range(self.Nb)]
        self.get_enc_bits_pc = [self.get_enc_bits(x) for x in range(self.Nb)]
        self.get_next_state_pc = [self.get_next_state(x) for x in range(self.Nb)]
        self.get_prev_state_pc = [self.get_prev_state(x) for x in range(self.Nb)]
        self.get_next_branches_pc = [self.get_next_branches(x) for x in range(self.Ns)]
        self.get_prev_branches_pc = [self.get_prev_branches(x) for x in range(self.Ns)]


# -------------------- helper functions ---------------------------------

def grouped(seq, n):
    it = iter(seq)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            # StopIteration
            return
        yield chunk


def dec2bin(val, k):
    bin_val = []
    for j in range(k):
        bin_val.append(val & 1)
        val = val >> 1
    return bin_val


def bin2dec(bin_val):
    n = len(bin_val)
    int_val = 0
    for j in range(n):
        int_val = int_val + bin_val[j] * 2 ** j
    return int_val


flatten = lambda l: (item for sublist in l for item in sublist)
