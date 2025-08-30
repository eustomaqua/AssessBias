# coding: utf-8

from fair_int_anal import PlotA_fair_ens


def test_fair_int_anal():
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
