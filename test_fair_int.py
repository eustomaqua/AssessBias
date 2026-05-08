# coding: utf-8

from fair_int_anal import PlotA_fair_ens
from fair_int_anal import PlotB_fair_ens
import pdb


def excl_test_fair_int_anal():
    kws = {}
    kws['exp'] = 'KF_exp1b'
    kws['pre'] = 'min_max'
    filename = f"{kws['exp'][:-1]}_iter5_pms"

    case = PlotA_fair_ens()
    filename = 'mCV_exp1_iter5_pms'
    sheetname = 'exp1b_minmax'
    raw_df = case.load_raw_dataset(filename, sheetname)

    case.schedule_mspaint(raw_df, sheetname)
    case.schedule_mspaint_avg(raw_df, sheetname)
    return


def test_fair_int_anal():
    filename = 'KF_exp1_iter5_pms'
    sheetname = 'exp1b_minmax'
    case = PlotB_fair_ens()
    raw_df = case.load_raw_dataset(filename, sheetname)
    case.schedule_mspaint(raw_df, sheetname)
    # case.schedule_mspaint_avg(raw_df, sheetname)
    return


def excl_test_fair_initial():
    from pyfair.datasets import (
        # Ricci, Adult, PropublicaRecidivism,
        # PropublicaViolentRecidivism,
        German, preprocess)
    from pyfair.preprocessing_dr import (
        adversarial, transform_X_and_y, transform_unpriv_tag)
    # from pyfair.preprocessing_hfm import (
    #     adverse_perturb, transform_X_A_and_y)

    # dt = Ricci()
    dt = German()
    # dt = Adult()
    # dt = PropublicaRecidivism()
    # dt = PropublicaViolentRecidivism()

    df = dt.load_raw_dataset()
    processed_dat = preprocess(dt, df)
    disturbed_dat = adversarial(dt, df, ratio=.97)
    pos_label = dt.get_positive_class_val('')
    non_sa, tmp = transform_unpriv_tag(dt, processed_dat['original'], 'both')

    processed_dat = processed_dat['numerical-binsensitive']
    disturbed_dat = disturbed_dat['numerical-binsensitive']
    X, y = transform_X_and_y(dt, processed_dat)
    Xp, _ = transform_X_and_y(dt, disturbed_dat)
    y[y != pos_label] = 0  # y[y == 2] = 0  # only for German

    print([sum(i).tolist() for i in non_sa + tmp])
    print(y.shape[0], X.shape[1], Xp.shape[1])
    pdb.set_trace()
    return
