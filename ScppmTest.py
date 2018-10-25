import numpy as np
from numpy.random import rand, randn
import Scppm
from turpy.Trellis import Trellis
from turpy.Interleaver import Interleaver
from turpy.ConvSISO import ConvSISO
from ScppmDecoder import ScppmDecoder

n_data = 7600 #7558

# create trellises, encoders and decoders instances
gp = [[1, 0, 1], [1, 1, 1]]
trellis = Trellis(gp)

# ppm modulator
m= 16
b= 0
ppm_mod = Scppm.PpmModulator(m, b)

# sisos
ppm_trellis = Scppm.PpmTrellis(ppm_mod)
inner_siso = Scppm.ConvSISOMultiBit(ppm_trellis)

outer_siso = ConvSISO(trellis)
outer_siso.backward_init = True

# create interleaver instances
len_e = trellis.r*n_data + trellis.r*(trellis.K-1)
print(len_e)
n_i = len_e #int(2**(np.ceil(np.log2(len_e)/2)*2))
il = Interleaver()
assert (n_i % ppm_mod.m_b) == 0
#assert n_i == 15120
#il.gen_qpp_perm_poly(n_i, 11, 210)
il.gen_rand_perm(n_i)

# create decoder
scppm_decoder = ScppmDecoder(il, inner_siso, outer_siso)
scppm_decoder.iterations = 20

# generate data
data_u = list(((rand(n_data) >= 0.5).astype(int)))

# encode
scppm_enc = Scppm.ScppmEncoder(trellis, il,ppm_mod)
[c, len_x, n_i2] = scppm_enc.encode(data_u)
assert n_i == n_i2

# noise
#EbNodB = 2.6
#EbNo = 10.0 ** (EbNodB / 10.0)
#print(EbNo)
#noise_std = float(np.sqrt(2) / np.sqrt(2 * EbNo))
#y = [16 * x + noise_std * float(randn(1)) for x in c]

# poisson noise
lambda_n = 0.2
lambda_s_dB = 2.8
lambda_s = 10.0 ** (lambda_s_dB / 10.0)
y=[]
for x in c:
    if x>0:
        ys = np.random.poisson(lambda_s)
    else:
        ys = 0
    yn = np.random.poisson(lambda_n)
    y.append(ys+yn)

ll=[]
for x in y:
    lh=(((lambda_s+lambda_n)**x) * np.exp(-lambda_s) ) / (lambda_n**x)
    ll.append(np.log(lh))

y_sparse = []
for ppm_sym in Scppm.grouped(y, (ppm_mod.m+ppm_mod.b)):
    index_max=ppm_sym.index(max(ppm_sym))
    empty_symbol = [0]*(ppm_mod.m+ppm_mod.b)
    empty_symbol[index_max]= ppm_sym[index_max]
    y_sparse.extend(empty_symbol)

# decode
[u, ber_iter] = scppm_decoder.decode(ll, n_data, n_i, len_x, data_u)

print("ber_iter= " + str(ber_iter))



