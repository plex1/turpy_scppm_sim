#! /usr/bin/env python
# title           : Scppm.py
# description     : This file implements scppm encoders and decoders
# author          : Felix Arnold
# python_version  : 3.5.2

import numpy as np
from turpy import utils
from turpy.ConvEncoder import ConvEncoder
from turpy.Trellis import Trellis
from turpy.ConvTrellisDef import ConvTrellisDef


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
        return list(utils.flatten(self.modulate_symbol(utils.bin2dec(x)) for x in utils.grouped(data, self.m_b)))

    def demodulate(self, symbol):
        return symbol.index(max(symbol[0:self.m]))

    def demodulate_stream(self, e):
        return list(
            utils.flatten(utils.dec2bin(self.demodulate(x), self.m_b) for x in utils.grouped(e, self.m + self.b)))


class ScppmEncoder(object):

    def __init__(self, trellis, interleaver, modulator):
        self.state = 0
        self.trellis = trellis
        self.interleaver = interleaver
        self.modulator = modulator

    def accu(self, d):
        trellis_accu = Trellis(ConvTrellisDef([[1, 0]], [0, 1]))
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


class AppmTrellis(object):

    def __init__(self, modulator):
        self.Ns = 2  # number of states
        self.Nb = 2 * modulator.m  # number of branches
        self.wc = modulator.m
        self.rsc = 0
        self.wu = int(np.log2(modulator.m))
        self.mod = modulator
        self.get_dat_precalc = []
        self.get_enc_bits_precalc = []

    def get_next_state(self, branch):
        return int((branch & 2 ** (self.wu - 1)) != 0)

    def get_prev_state(self, branch):
        return int((branch & 2 ** (self.wu)) != 0)

    def get_prev_branches(self, state):
        return np.array([x + state * 2 ** (self.wu - 1) for x in range(2 ** (self.wu - 1))] + [
            x + state * 2 ** (self.wu - 1) + 2 ** self.wu for x in range(2 ** (self.wu - 1))])

    def get_next_branches(self, state):
        return np.array([x + state * 2 ** (self.wu) for x in range(2 ** (self.wu))])

    def get_next_branch(self, state, dat):
        print("not implemented")

    def get_enc_bits(self, branch):
        return self.mod.modulate_symbol(branch & (2 ** self.wu - 1))

    def get_dat(self, branch):
        return utils.dec2bin(
            (branch & (2 ** self.wu - 1)) ^ (((branch & (2 ** (self.wu - 1) - 1)) << 1) + self.get_prev_state(branch)),
            self.wu)
