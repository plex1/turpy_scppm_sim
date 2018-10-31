import numpy as np
from numpy.random import rand, randn
import Scppm
from turpy.Trellis import Trellis
from turpy.ConvTrellisDef import ConvTrellisDef
from turpy.Interleaver import Interleaver
from turpy.SisoDecoder import SisoDecoder
from ScppmDecoder import ScppmDecoder
from turpy import utils


def main():
    # settings
    ccsds = True
    if not ccsds:
        n_data = 4000  # variable
    else:
        n_data = 7560  # fixed
    sparse = False
    llh_calc = True
    poisson = True

    # create ppm modulator
    m = 16
    b = 0
    ppm_mod = Scppm.PpmModulator(m, b)

    # create trellises, encoders and decoders instances
    gp = [[1, 0, 1], [1, 1, 1]]
    outer_trellis = Trellis(ConvTrellisDef(gp))
    appm_trellis = Trellis(Scppm.AppmTrellis(ppm_mod))

    # create sisos
    inner_siso = SisoDecoder(appm_trellis)
    inner_siso.backward_init = False
    outer_siso = SisoDecoder(outer_trellis)

    # create interleaver
    len_e = int(outer_trellis.get_rate() * n_data)
    n_i = len_e  # set the interleaver size to the length of the encoded stream
    il = Interleaver()
    assert (n_i % ppm_mod.m_b) == 0
    if ccsds:
        assert n_i == 15120
        il.gen_qpp_perm_poly(n_i, 11, 210)
    else:
        il.gen_rand_perm(n_i)

    # create decoder
    scppm_decoder = ScppmDecoder(il, inner_siso, outer_siso)
    scppm_decoder.iterations = 30

    # generate data
    data_u = list(((rand(n_data) >= 0.5).astype(int)))
    data_u[-1:-2] = [0] * 2  # zero termination

    # encode
    scppm_enc = Scppm.ScppmEncoder(outer_trellis, il, ppm_mod)
    [c, len_x, n_i2] = scppm_enc.encode(data_u)
    assert n_i == n_i2

    # insert noise
    if poisson:
        # poisson noise
        lambda_n = 0.2
        lambda_s_dB = 2.8
        lambda_s = 10.0 ** (lambda_s_dB / 10.0)
        y = []
        for x in c:
            if x > 0:
                ye = np.random.poisson(lambda_s + lambda_n)
            else:
                ye = np.random.poisson(lambda_n)
            y.append(ye)
    else:  # awgn
        EbNodB = 2.4
        EbNo = 10.0 ** (EbNodB / 10.0)
        # noise_std = float((outer_trellis.r * appm_trellis.wb/(ppm_mod.m+ppm_mod.b)) / (2 * EbNo)) # optical square channel
        noise_std = float(np.sqrt(2 + 1) / np.sqrt(2 * EbNo))
        y = [2 * x + noise_std * float(randn(1)) for x in c]

    # transform to log likelihood
    if llh_calc:
        y_llh = []
        for x in y:
            lh = (((lambda_s + lambda_n) ** x) * np.exp(-lambda_s)) / (lambda_n ** x)
            y_llh.append(np.log(lh))
            y = y_llh

    # remove some information
    if sparse:
        y_sparse = []
        for ppm_sym in utils.grouped(y, (ppm_mod.m + ppm_mod.b)):
            index_max = ppm_sym.index(max(ppm_sym))
            symbol = [0] * (ppm_mod.m + ppm_mod.b)
            symbol[index_max] = ppm_sym[index_max]
            y_sparse.extend(symbol)
        y = y_sparse

    # decode
    [u, ber_iter] = scppm_decoder.decode(y, n_data, data_u)

    # result
    print("ber_iter= " + str(ber_iter))


if __name__ == "__main__":
    main()
