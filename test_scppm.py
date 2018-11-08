import Scppm
import utils


def test_appm_trellis_def():

    # create ppm modulator
    m = 16
    b = 0
    ppm_mod = Scppm.PpmModulator(m, b)

    # created appm trellis definition
    atd= Scppm.AppmTrellisDefB(ppm_mod)

    #get_dat
    assert [0, 0, 0, 0] == atd.get_dat(0)
    assert [0, 0, 0, 0] == atd.get_dat(1)
    assert [1, 0, 0, 0] == atd.get_dat(2)
    assert [0, 0, 0, 1] == atd.get_dat(16)

    assert utils.bin2dec([0, 1, 1, 1]) == atd.get_a(utils.bin2dec([0, 0, 1, 0, 0]))
    assert utils.bin2dec([0, 1, 0, 0]) == atd.get_a(utils.bin2dec([0, 0, 1, 1, 0]))

    assert utils.bin2dec([1, 0, 1, 0]) == atd.get_a(utils.bin2dec([0, 1, 1, 1, 1]))
    assert [1, 1, 1, 1] == atd.get_dat(utils.bin2dec([0, 1, 1, 1, 1]))

    assert utils.bin2dec([0, 1, 0, 1]) == atd.get_a(utils.bin2dec([1, 1, 1, 1, 1]))
    assert [1, 1, 1, 1] == atd.get_dat(utils.bin2dec([0, 1, 1, 1, 1]))

    assert 1 == atd.get_next_state(utils.bin2dec([0, 1, 0, 0, 0]))
    assert 0 == atd.get_next_state(utils.bin2dec([0, 1, 0, 0, 1]))
    assert 1 == atd.get_next_state(utils.bin2dec([0, 1, 0, 1, 1]))


    assert ppm_mod.modulate_symbol(14) == atd.get_enc_bits(7)
    assert ppm_mod.modulate_symbol(7) == atd.get_enc_bits(18)

