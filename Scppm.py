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
        enc=[0]*(self.m+self.b)
        enc[symbol_num]=1
        return enc

    def modulate_stream(self, data):
        return list(flatten(self.modulate_symbol(bin2dec(x)) for x in grouped(data, self.m_b)))

    def demodulate(self, symbol):
        return symbol.index(max(symbol[0:self.m]))

    def demodulate_stream(self, e):
        return list(flatten(dec2bin(self.demodulate(x), self.m_b) for x in grouped(e, self.m+self.b)))


class ScppmEncoder(object):

    def __init__(self, trellis, interleaver, modulator):
        self.state = 0
        self.trellis = trellis
        self.interleaver = interleaver
        self.modulator = modulator
        self.r = 3

    def accu(self, d):
        trellis_accu = Trellis([[1, 0]],[0, 1] )
        cve = ConvEncoder(trellis_accu)
        return cve.encode(d, False)


    def encode(self, d):
        self.cve = ConvEncoder(self.trellis)
        #print(len(d))
        d = self.cve.encode(d)  # encoding incl zero termination
        len_x= len(d)
        #print(len(d))
        #d=d+[0]*(int(2**(np.ceil(np.log2(len(d))/2)*2))-len(d))
        d = d + [0]*(len(self.interleaver.perm) - len(d))
        #print(len(d))
        d = self.interleaver.interleave(d) # interleaving
        N_i = len(d)
        #print("after interleaver: " + str(d))
        d = self.accu(d)
        #print(len(d))
        e = self.modulator.modulate_stream(d) # modulation
        #print(len(e))
        return [e ,len_x, N_i]


import numpy as np


class PpmTrellis(object):

    def __init__(self, modulator):

        self.Ns = 2  # number of states
        self.Nb = 2*modulator.m  # number of branches
        self.r = modulator.m
        self.rsc = 0
        self.wb = int(np.log2(modulator.m))
        self.mod = modulator
        self.get_dat_precalc = []
        self.get_enc_bits_precalc = []
        self.optimize = False
        self.precalc()

    def get_rate(self):
        return self.gen_matrix.shape[0]

    def get_k(self):
        return self.gen_matrix.shape[1]

    def get_next_state(self, branch):
        return int((branch & 2**(self.wb-1)) != 0)

    def get_prev_state(self, branch):
        return int((branch & 2 ** (self.wb)) != 0)

    def get_prev_branches(self, state):
        return np.array([x+ state*2**(self.wb-1) for x in range(2**(self.wb-1))] + [x+state*2**(self.wb-1) + 2**self.wb for x in range(2**(self.wb-1))])


    def get_next_branches(self, state):
        return np.array([x + state * 2 ** (self.wb) for x in range(2 ** (self.wb))]  )

    def get_next_branch(self, state, dat):
        print("not implemented")

    def get_enc_bits(self, branch):
        return self.mod.modulate_symbol(branch & (2**self.wb-1))

    def get_dat(self, branch):
        return self._dec2bin((branch & (2 ** self.wb - 1)) ^ ( ((branch & (2**(self.wb-1)-1)) << 1 ) + self.get_prev_state(branch)), self.wb)

    def _dec2bin(self, val, k):
        bin_val = []
        for j in range(k):
            bin_val.append(val & 1)
            val = val >> 1
        return bin_val

    def precalc(self):
        self.get_dat_pc=[]
        self.get_enc_bits_pc = []
        for i in range(self.Nb):
            self.get_dat_pc.append(self.get_dat(i))
            self.get_enc_bits_pc.append(self.get_enc_bits(i))

        self.get_dat_pc = [self.get_dat(x) for x in range(self.Nb)]
        self.get_enc_bits_pc = [self.get_enc_bits(x) for x in range(self.Nb)]
        self.get_next_state_pc = [self.get_next_state(x) for x in range(self.Nb)]
        self.get_prev_state_pc = [self.get_prev_state(x) for x in range(self.Nb)]
        self.get_next_branches_pc = [self.get_next_branches(x) for x in range(self.Ns)]
        self.get_prev_branches_pc = [self.get_prev_branches(x) for x in range(self.Ns)]

        self.optimize = True

class ConvSISOMultiBit(object):

    def __init__(self, trellis):
        self.state = 0
        self.trellis = trellis
        self.remove_tail = True
        self.forward_init = True
        self.backward_init = False

    def decode(self, ys, yp, la, n_data):

        trellis = self.trellis
        n_stages = int(n_data/self.trellis.wb)
        sm_vec_init = [0] + [-10 * self.forward_init] * (trellis.Ns - 1)  # init state metric vector

        # forward (alpha)
        sm_vec = sm_vec_init
        sm_forward = []
        for i in range(0, n_stages):  # for each stage
            sm_vec_new = []
            llr = yp[trellis.r * i:trellis.r * (i + 1) ]
            ysp = ys[trellis.wb * i: trellis.wb * (i+1)]
            lap = la[trellis.wb * i: trellis.wb * (i+1)]
            for j in range(trellis.Ns):  # for each state
                branches = trellis.get_prev_branches_pc[j]
                sums = []
                for k in range(len(branches)):  # for each branch
                    branch_metric = 0
                    for l in range(trellis.r):  # for each encoded bit
                        if trellis.get_enc_bits_pc[branches[k]][l] == 1:
                            branch_metric += llr[l]
                    for l in range(trellis.wb):  # for each data bit
                        if trellis.get_dat_pc[branches[k]][l]:
                            branch_metric += ysp[l] + lap[l]
                    sums.append(sm_vec[trellis.get_prev_state_pc[branches[k]]] + branch_metric)  # add
                sm_vec_new.append(max(sums))  # compare and select
            sm_vec = list(sm_vec_new)
            sm_forward.append(sm_vec)

        # backward (beta)
        sm_backward = []
        lu = []
        sm_vec = [0] + [-10 * self.backward_init] * (
                trellis.Ns - 1)  # init state metric vector # init state metric vector
        for i in reversed(range(0, n_stages)):  # for each stage
            sm_vec_new = []
            llr = yp[trellis.r * i : trellis.r * (i + 1)]
            ysp = ys[trellis.wb*i: trellis.wb*(i+1)]
            lap = la[trellis.wb * i: trellis.wb * (i+1)]
            max_branch = [[-10, -10] for i in range(trellis.wb)]
            for j in range(trellis.Ns):  # for each state
                branches = trellis.get_next_branches_pc[j]
                sums = []
                for k in range(len(branches)):  # for each branch
                    branch_metric = 0
                    for l in range(trellis.r):  # for each encoded bit
                        if trellis.get_enc_bits_pc[branches[k]][l] == 1:
                            branch_metric += llr[l]
                    for l in range(trellis.wb):  # for each data bit
                        if trellis.get_dat_pc[branches[k]][l]:
                            #print(ysp)
                            branch_metric += ysp[l] + lap[l]
                    branch_sum = sm_vec[trellis.get_next_state_pc[branches[k]]] + branch_metric  # add (gamma)
                    sums.append(branch_sum)

                    if i == 0:
                        post_branch_metric = branch_sum + sm_vec_init[j]
                    else:
                        post_branch_metric = branch_sum + sm_forward[i - 1][j]

                    # soft output calculation
                    out = trellis.get_dat_pc[branches[k]]
                    #if branch_sum>0:
                    #    print("out: "+ str(out) + ", k: "+ str(k)+ ", stage i: "+ str(i)+ ", max_branch: "+ str(max_branch)+ ", pbm:"+ str(post_branch_metric)+ ", smf: " + str(sm_forward[i - 1][j]))
                    for n in range(trellis.wb):
                        if post_branch_metric > max_branch[n][out[n]]:
                            #print("mb before: " +str(max_branch) + " ,n: "+ str(n)+", outn "+ str(out[n]))
                            max_branch[n][out[n]] = post_branch_metric
                            #print("mb after: " +str(max_branch))
                    #if branch_sum>0:
                        #print( "max_branch: "+ str(max_branch))

                sm_vec_new.append(max(sums))  # compare and select

            sm_vec = list(sm_vec_new)
            sm_backward.insert(0, sm_vec)
            if i < n_data or not self.remove_tail:  # soft output
                for n in reversed(range(trellis.wb)):
                    lu.append(max_branch[n][1] - max_branch[n][0])

        lu = list(reversed(lu))
        return lu

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