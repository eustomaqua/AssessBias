# coding: utf-8


import os
import pdb
import pandas as pd
import numpy as np

from pyfair.utils_empirical import GraphSetup, GRP_FAIR_COMMON
from pyfair.facil.utils_const import unique_column, DTY_FLT

from pyfair.granite.draw_addtl import (
    single_line_reg_with_distr, multi_lin_reg_without_distr,
    # lineplot_with_uncertainty,  # scatter_with_marginal_distrib,
    line_reg_with_marginal_distr)  # multi_lin_reg_with_distr,
from pyfair.granite.draw_fancy import (
    multi_boxplot_rect, radar_chart)  # boxplot_rect,
from pyfair.granite.draw_chart import (
    analogous_confusion_extended)  # ,multiple_scatter_chart)

# from pyfair.granite.draw_fancy import (
#     multi_boxplot_rect_revised, tabular_chart,
#     radar_chart_gather, tabular_chart_gather)
from pyfair.granite.draw_chart import anal_conf_extended_subplt
# from pyfair.granite.draw_addtl import (
#     linreg_w_marg_dist_revised,  # linreg_w_marg_dist_revised_subpltv1,
#     gathering_lin_reg_sup_tim)
from pyfair.granite.draw_fancy import multi_boxplot_rect_revised
from cont_draw import (radar_chart_gather, tabular_chart_gather,
                       linreg_w_marg_dist_rev_sup_pv1,
                       linreg_w_marg_dist_rev_sup_pv2,
                       gathering_lin_reg_sup_tim,
                       )  # linreg_w_marg_dist_revised)
# from pyfair.granite.draw_addtl import linreg_w_marg_dist_revised


# -----------------------------
# Exp1: bin-val vs. multi-val
#
# Plot 1:
# Plot 2:
#


class PlotA_initial(GraphSetup):
    # pass
    _dr_ptb = 'K'  # perturb(ation)  # 'K','L'

    _perf_metric = [
        'Accuracy', 'Precision', 'Recall',  # 'Sensitivity'
        'Specificity', r'$\mathrm{f}_1$ score',  # r'$\bar{g}$',
        # 'G mean', 'bal. acc', 'discr power']  # 'discrim. discrimn.'
        'g mean', 'bal. acc', 'discr power']
    # _dal_metric = [
    #     r'$\Delta(\text{Accuracy})$', r'$\Delta(\text{Precision})$',
    #     r'$\Delta(\text{Recall})$', r'$\Delta(\text{Specificity})$',
    #     r'$\Delta(\mathrm{f}_1 ~\text{score})$', r'$\Delta(\bar{g})$',
    #     r'$\Delta(\text{bal. acc})$', r'$\Delta(\text{discr power})$']
    _dal_metric = [
        r'$\Delta$(Accuracy)', r'$\Delta$(Precision)',
        r'$\Delta$(Recall)', r'$\Delta$(Specificity)',
        r'$\Delta$($\mathrm{f}_1$ score)',  # r'$\Delta(\bar{g})$',
        # r'$\Delta$(G mean)',  # r'$\Delta$(g-mean)', 'g_mean'
        r'$\Delta$(g mean)',
        r'$\Delta$(bal. acc)', r'$\Delta$(discr power)']

    def obtain_tag_col(self, tag='tst'):
        csv_row_1 = unique_column(12 + 158 * 2)
        tag_trn = csv_row_1[12: 12 + 158]
        tag_tst = csv_row_1[-158:]
        tag_col = tag_trn if tag == 'trn' else tag_tst

        # sub-tags
        st_acc = tag_col[: 24]
        st_grp = [tag_col[24: 24 + 29], tag_col[53: 24 + 58]]
        st_dr = tag_col[24 + 58: 24 + 58 + 18]  # 24+76=100
        st_hfm_drt = tag_col[100: 100 + 29]
        st_hfm_app = tag_col[100 + 29: 100 + 58]

        tag_common = st_acc[:8] + st_acc[-8:] + st_dr[:4] + [
            st_dr[6], st_dr[9], st_dr[12]] + st_dr[-3:] + st_hfm_drt[
            8:15] + st_hfm_app[8:15]  # delta, GEI alph=.2|.5|.8,Theil
        tag_sa1 = st_grp[0][6:22] + st_grp[0][-1:] + st_hfm_drt[
            :4] + st_hfm_drt[15:22] + st_hfm_app[:4] + st_hfm_app[
            15:22] + st_grp[0][-7: -1]
        tag_sa2 = st_grp[1][6:22] + st_grp[1][-1:] + st_hfm_drt[
            4:8] + st_hfm_drt[22:29] + st_hfm_app[4:8] + st_hfm_app[
            22:29] + st_grp[1][-7: -1]

        # tag_common: perf 8+ delta(perf) 8+ dr(loss,ut,hat_bias,ut) 4+
        #             GEI(alph=.2|.5|.8, Theil,T(Theil),T(GEIx11)) 6+
        #             hfm direct multiver 7+ hfm approx multiver 7
        # tag_sa1/2 : DP,EO,PQP,tim, SP-ext *3, SP-ext-avg *3,
        #             SP-ext-meticulous *3, SP-ext-avg-meticulous *3,tim,
        #             hfm drt (bin 4+ nonbin 7), hfm app (bin 4+ nonbin 7)
        # siz: 16+4+6+14=40, 4+13+11*2=17+22=39
        # return tag_common, tag_sa1, tag_sa2

        # return tag_common + csv_row_1[10:12], tag_sa1, tag_sa2
        tag_common = tag_common + csv_row_1[10:12]
        tag_common.extend([st_acc[8 + 2], st_acc[8 + 3], ])
        return tag_common, tag_sa1, tag_sa2

    def obtain_binval_senatt(self, dframe, id_set,  # nb_set,
                             tag='tst'):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(tag)
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[1] + 1: id_set[2]][tag_acc + tag_sa1]
        df_raw[self._dr_ptb] = dframe.iloc[id_set[1]][self._dr_ptb]
        for k in [1, 2]:
            df_tmp = dframe.iloc[id_set[
                k] + 1: id_set[k + 1]][tag_acc + tag_sa2]
            df_tmp = df_tmp.rename(columns=columns)
            df_tmp[self._dr_ptb] = dframe.iloc[id_set[k]][self._dr_ptb]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)

        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[
                k] + 1: id_set[k + 1]][tag_acc + tag_sa1]
            df_tmp[self._dr_ptb] = dframe.iloc[id_set[k]][self._dr_ptb]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw

    def obtain_multival_senatt(self, dframe, id_set,  # nb_set,
                               tag='tst', first_incl=False):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(tag)
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        df_raw = dframe.iloc[id_set[2] + 1: id_set[3]][tag_acc + tag_sa1]
        df_raw[self._dr_ptb] = dframe.iloc[id_set[2]][self._dr_ptb]
        for k in [3, 4]:
            df_tmp = dframe.iloc[id_set[k] + 1: id_set[
                k + 1]][tag_acc + tag_sa2]
            df_tmp = df_tmp.rename(columns=columns)
            df_tmp[self._dr_ptb] = dframe.iloc[id_set[k]][self._dr_ptb]
            df_raw = pd.concat([df_raw, df_tmp], axis=0)
        # df_raw = df_raw.reset_index(drop=True)
        # np.isnan(df_raw.values.astype('float')).any()
        if not first_incl:  # if not first: first_incl.
            return df_raw
        df_tmp = dframe.iloc[id_set[0] + 1: id_set[1]][tag_acc + tag_sa1]
        df_tmp[self._dr_ptb] = dframe.iloc[id_set[0]][self._dr_ptb]
        df_raw = pd.concat([df_raw, df_tmp], axis=0)
        return df_raw

    def draw_extended_grp_tim(self, df, tag_X, tag_Ys, figname,
                              verbose=False):  # annot_X, annot_Ys,
        X = df[tag_X].values.astype(DTY_FLT)
        Ys = [df[i].values.astype(DTY_FLT) / X for i in tag_Ys]
        antX = r'T_\text{bin-val}'  # refined
        antYs = [r'T_\text{multival}', r'T_\text{multival alt}']
        antYs.extend([r'T_\text{extGrp}', r'T_\text{alt.extGrp}'])

        annots = [f'${antX}$ (sec)', f'${antYs[0]}$',
                  f'${antYs[0]}={antX}$']
        kws = {'linreg': True, 'snspec': 'sty6'}  # 'sty4'
        annots[1] = r'$\frac{T_\text{multival}}{T_\text{bin-val}}-1$'
        single_line_reg_with_distr(X, Ys[0] - 1., annots,
                                   f'{figname}_tim_st6a', **kws)
        annots[1] = r'$\lg(\frac{T_\text{multival}}{T_\text{bin-val}})$'
        single_line_reg_with_distr(X, np.log10(Ys[0]), annots,
                                   f'{figname}_tim_st6b', **kws)

        # annots[2] = f'${antYs[1]}={antX}$'
        # annots[1] = r'$\frac{T_\text{multival alt}}{T_\text{bin-val}}$'
        if not verbose:
            return
        tmp = [[r'$\frac{ T_\text{extGrp} }{ T_\text{bin-val} }-1$',
                r'$\lg(\frac{ T_\text{extGrp} }{ T_\text{bin-val} })$'], [
               r'$\frac{ T_\text{alt.extGrp} }{ T_\text{bin-val} }-1$',
               r'$\lg(\frac{ T_\text{alt.extGrp} }{ T_\text{bin-val} })$']]
        for i in [-2, -1]:
            annots_sep = [f'${antX}$ (sec)', '', f'${antYs[i]}={antX}$']
            annots_sep[1] = tmp[i][0]
            single_line_reg_with_distr(
                X, Ys[i] - 1, annots_sep,
                f'{figname}_grptim_sep{i}_st6a', **kws)
            annots_sep[1] = tmp[i][1]
            single_line_reg_with_distr(
                X, np.log10(Ys[i]), annots_sep,
                f'{figname}_grp_tim_sep{i}_st6b', **kws)
        return

    def draw_extended_hfm_tim(self, df, tag_X, tag_Ys, figname):
        X = df[tag_X].values.astype(DTY_FLT)  # T(drt bin-val)
        Ys = [df[i].values.astype(DTY_FLT) for i in tag_Ys]
        # T(drt multi-val), T(app bin-val), T(app multi-val)
        # annots = [r'$T_\text{hfm bin-val}$', r'$T_\text{hfm multival}$',
        #           r'$T_\text{hfm multival}=T_\text{hfm bin-val}$']
        annots = [r'$T_{\mathbf{df}_\text{prev} ~\text{(bin-val)}}$',
                  r'$T_{\mathbf{df} ~\text{(multival)}}$',
                  r'$T_{\mathbf{df} ~\text{(multival)}}=T_{\mathbf{df}_\text{prev} ~\text{(bin-val)}}$']
        antZs = [r'$T_{\mathbf{df} ~\text{(multival)}}$', r'$T_{\hat{\mathbf{df}} ~\text{(bin-val)}}$', r'$T_{\hat{\mathbf{df}} ~\text{(multival)}}$']
        # multi_lin_reg_without_distr(X, Ys, antZs, annots,
        #                             f'{figname}_tim_sty4', snspec='sty4')

        n_ell = len(tag_Ys)
        Zs = [i / X - 1. for i in Ys]
        annots[1] = r'$\frac{ T_{\mathbf{df} ~\text{(multival)}} }{ T_{\mathbf{df}_\text{prev} ~\text{(bin-val)}} }-1$'
        multi_lin_reg_without_distr(
            X, Zs, antZs[:n_ell], annots,
            f'{figname}_tim_st6a', snspec='sty6')
        kws = {'linreg': True, 'snspec': 'sty6'}
        single_line_reg_with_distr(
            X, Zs[0], annots, f'{figname}_prtim_6a', **kws)
        Zs = [np.log10(i / X) for i in Ys]
        annots[1] = r'$\lg(\frac{ T_{\mathbf{df} ~\text{(multival)}} }{ T_{\mathbf{df}_\text{prev} ~\text{(bin-val)}} })$'
        multi_lin_reg_without_distr(
            X, Zs, antZs[:n_ell], annots,
            f'{figname}_tim_st6b', snspec='sty6')
        single_line_reg_with_distr(
            X, Zs[0], annots, f'{figname}_prtim_6b', **kws)
        return

    def draw_extended_grp_scat(self, df, tag_grp, tag_ext,
                               tag_ext_alt, figname,
                               verbose=False):
        # labels = ['grp', 'extGrp', 'alt.extGrp']  # 'extAlt'
        labels = ['ori', 'ext', 'alt']
        lbl_hfm = [[r'$\mathbf{df}_\text{prev}$', r'$\mathbf{df}$',
                    r'$\mathbf{df}^{avg}$'], [
            r'$\hat{\mathbf{df}}_\text{prev}$', r'$\hat{\mathbf{df}}$',
            r'$\hat{\mathbf{df}}^{avg}$'], ]
        lbl_dim2 = ['DP', 'EO', 'PQP', r'$\mathbf{df}_\text{prev}$',
                    r'$\hat{\mathbf{df}}_\text{prev}$']
        lbl_dim2[2] = 'PP'
        lbl_dim2[:3] = GRP_FAIR_COMMON
        multi_boxplot_rect(
            df, tag_grp[:3], tag_ext[:3],
            figname=f'{figname}_grpext', annotX=lbl_dim2[:3],
            locate="upper left")
        multi_boxplot_rect(
            df, tag_grp[:3], tag_ext[:3], tag_ext_alt[:3],
            figname=f'{figname}_grpalt', annotX=lbl_dim2[:3],
            locate="upper left")
        if not verbose:
            return

        for i, tg in enumerate(tag_grp):
            # data = [df[tg].values.astype(DTY_FLT),
            #         df[tag_ext[i]].values.astype(DTY_FLT),
            #         df[tag_ext_alt[i]].values.astype(DTY_FLT)]
            fgn = '{}_{}'.format(
                figname, f'grp{i+1}' if i < 3 else f'hfm{i+3}')
            # boxplot_rect(data, labels, fgn + '_prim')
            multi_boxplot_rect(df, [tg, tag_ext[
                i], tag_ext_alt[i]], figname=fgn,
                annotX=labels if i < 3 else lbl_hfm[i - 3])  # not tag_Xs
        multi_boxplot_rect(df, tag_grp, tag_ext,
                           figname=f'{figname}_dim2', annotX=lbl_dim2)
        multi_boxplot_rect(df, tag_grp, tag_ext, tag_ext_alt,
                           figname=f'{figname}_dim3', annotX=lbl_dim2)
        return

    def obtain_sing_dat_cls(self, pick_set, pick_clf,
                            tag_acc, tag_sa1, tag_sa2,
                            dframe, id_set,  # tag='tst',
                            multival=True):  # nonbin=True):
        columns = {t2: t1 for t1, t2 in zip(tag_sa1, tag_sa2)}
        picked_a = id_set[pick_set] + 1 + pick_clf * self._nb_cv
        picked_b = id_set[pick_set] + 1 + (pick_clf + 1) * self._nb_cv
        if multival:
            assert pick_set in [0, 2, 3, 4]
            df_tmp = dframe.iloc[picked_a: picked_b]
            if pick_set in [3, 4]:
                df_tmp = df_tmp[tag_acc + tag_sa2]
                df_tmp = df_tmp.rename(columns=columns)
            else:
                df_tmp = df_tmp[tag_acc + tag_sa1]
        else:
            assert pick_set in [1, 2, 3, 4]
            df_tmp = dframe.iloc[picked_a: picked_b]
            if pick_set in [1, 3, 4]:
                df_tmp = df_tmp[tag_acc + tag_sa1]
            else:
                df_tmp = df_tmp[tag_acc + tag_sa2]
                df_tmp = df_tmp.rename(columns=columns)
            if pick_set == 1:
                df_alt = dframe.iloc[picked_a: picked_b][
                    tag_acc + tag_sa2].rename(columns=columns)
                df_tmp = pd.concat([df_tmp, df_alt], axis=0)
        return df_tmp

    def depict_separately(self, pick_set, pick_clf, df, id_set,
                          tag_mk='tst', fgn='', multival=True,
                          verbose=False):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(
            tag_mk)  # mark)
        tag_acc = tag_acc[: -2]
        df_alt = self.obtain_sing_dat_cls(
            pick_set, pick_clf, tag_acc, tag_sa1, tag_sa2,
            df, id_set, multival)
        sub_grp = tag_sa1[:3] + [tag_sa1[16 + 3], tag_sa1[27 + 3]]
        sub_ext = tag_sa1[4:10][:3] + [tag_sa1[16 + 7], tag_sa1[27 + 7]]
        sub_ext_alt = tag_sa1[10:16][:3] + [tag_sa1[
            16 + 10], tag_sa1[27 + 10]]
        sub_idv = tag_acc[16:16 + 4 + 6]  # dr 4+ GEI.alph 3+ Theil+Tx2
        sub_idv = [sub_idv[2], ] + sub_idv[4:-2]
        sub_idv = [sub_idv[0], sub_idv[2], sub_idv[-1]]

        currX = sub_grp[:3] + sub_idv + sub_grp[-2:]
        labels = ['DP', 'EO', 'PP', 'DR',  # r'GEI ($\alpha=0.5$)',
                  r'GEI ($\alpha$=0.5)',
                  'Theil', r'$\mathbf{df}_\text{prev}$',
                  r'$\hat{\mathbf{df}}_\text{prev}$']
        labels[:3] = GRP_FAIR_COMMON
        df_tmp = df_alt[currX]  # =df_alt[curr_X].mean(axis=0)
        # radar_chart(df_tmp, currX, annotX=labels, figname='test_b', clockwise=True)
        # radar_chart(df_tmp, currX, annotX=labels, figname='test_a')
        for i in currX:
            df_tmp.loc[:, i] = float(df_tmp[i].mean())
        radar_chart(df_tmp, currX, annotX=labels,
                    figname=f'{fgn}_s{pick_set}c{pick_clf}_ori',
                    clockwise=True)

        df_tmp = df_tmp.reset_index(drop=True)[:3]
        df_tmp_tmp = df_alt[sub_ext]
        for i, j in zip(sub_grp, sub_ext):
            df_tmp.loc[1, i] = float(df_tmp_tmp[j].mean())
        df_tmp_tmp = df_alt[sub_ext_alt]
        for i, j in zip(sub_grp, sub_ext_alt):
            df_tmp.loc[2, i] = float(df_tmp_tmp[j].mean())
        annotY = ['ori', 'ext', 'alt']  # 'ext.alt']
        if verbose:
            radar_chart(df_tmp[:2], currX, labels, annotY[:2],
                        figname=f'{fgn}_s{pick_set}c{pick_clf}_ext')
        radar_chart(df_tmp, currX, labels, annotY,
                    figname=f'{fgn}_s{pick_set}c{pick_clf}_extalt')
        return

    def draw_trade_off(self, df, pick, tag_X, tag_Ys, figname,
                       ver_mark=''):
        annotZs = GRP_FAIR_COMMON + [
            r'GEI ($\alpha$=0.5)', 'Theil', 'DR']
        # tmp_ext = ['{:6s} ext'.format(i) for i in annotZs[:3]]
        # tmp_ext_alt = ['{:6s} alt'.format(i) for i in annotZs[:3]]
        # tmp_ext[1] = f'{annotZs[1]} ext'
        # tmp_ext_alt[1] = f'{annotZs[1]} alt'  # not '{:7s}'
        tmp_ext = [r'$\text{DP}^\text{ext}$',
                   r'$\text{EOpp}^\text{ext}$',
                   r'$\text{PP}^\text{ext}$', ]
        tmp_ext_alt = [r'$\text{DP}^\text{alt}$',
                       r'$\text{EOpp}^\text{alt}$',
                       r'$\text{PP}^\text{alt}$', ]
        for pk in pick:
            annotX = self._perf_metric[pk]  # pick]
            # annots = (annotX, "Fairness")  # 'Fairness measure'
            # X = df[tag_X[pk]].values.astype(DTY_FLT)
            # Ys = df[tag_Ys[0][:3] + tag_X[-2:]].values.astype(DTY_FLT).T
            # multiple_scatter_chart(
            #     X, Ys, annots, annotZs, f'{figname}_to{pk}v',
            #     ind_hv='v', identity=False)
            # scatter_with_marginal_distrib(
            #     df, tag_X[pk], 'Fairness', tag_Ys[0][:3],
            #     GRP_FAIR_COMMON, annotX=annotX, annotY='Fairness',
            #     figname=f'{figname}_to{pk}_s4')
            # line_reg_with_marginal_distr(
            #     df, tag_X[pk], 'Fairness', tag_Ys[0][:3] + tag_X[
            #         -2:] + tag_X[-3:-2], annotZs, annotX=annotX,
            #     annotY='Fairness', snspec='sty4b',
            #     figname=f'{figname}_to{pk}_s4')
            line_reg_with_marginal_distr(  # tag_X[-2:]+tag_X[-3:-2]
                df, tag_X[pk], 'Fairness', tag_X[-3:], annotZs[-3:],
                annotX=annotX, annotY='Individual fairness',
                snspec='sty4b', figname=f'{figname}_to{pk}_s4')
            line_reg_with_marginal_distr(
                df, tag_X[pk], 'Fairness', tag_Ys[0][:3], annotZs[:3],
                annotX=annotX, annotY='Group fairness (bin-val)',
                snspec='sty4b', figname=f'{figname}_to{pk}_s1')
            line_reg_with_marginal_distr(
                df, tag_X[pk], 'Fairness', tag_Ys[1][:3],
                # [f'{i} ext.' for i in annotZs[:3]], annotX=annotX,
                tmp_ext, annotX=annotX,
                annotY='Extended group fairness (multival)',
                snspec='sty4b', figname=f'{figname}_to{pk}_s2')
            line_reg_with_marginal_distr(
                # df, tag_X[pk], 'Fairness', tag_Ys[0][:3],
                # [f'{i} ext. alt' for i in annotZs[:3]], annotX=annotX,
                df, tag_X[pk], 'Fairness', tag_Ys[2][:3],
                tmp_ext_alt, annotX=annotX,
                annotY='Alternative extended group fairness (multival)',
                snspec='sty4b', figname=f'{figname}_to{pk}_s3')

        key_A = [tag_X[:8][i] for i in pick]
        key_C = [tag_X[8:16][i] for i in pick]
        key_B_bin = tag_Ys[0][:3] + tag_X[-3:] + tag_Ys[0][-2:]
        key_B_nonbin = tag_Ys[1][:3] + tag_X[-3:] + tag_Ys[1][-2:]
        key_B_extalt = tag_Ys[2][:3] + tag_X[-3:] + tag_Ys[2][-2:]
        lbl_A = [self._perf_metric[i] for i in pick]
        lbl_C = [self._dal_metric[i] for i in pick]
        # lbl_B_bin = GRP_FAIR_COMMON + [
        #     'DR', r'GEI ($\alpha=0.5$)', 'Theil',
        #     r'$\mathbf{df}_\text{prev}$',
        #     r'$\hat{\mathbf{df}}_\text{prev}$']
        lbl_B_bin = annotZs + [r'$\mathbf{df}_\text{prev}$',
                               r'$\hat{\mathbf{df}}_\text{prev}$']
        # lbl_B_ext = [f'{i} ext.' for i in GRP_FAIR_COMMON] + lbl_B_bin[
        #     3:6] + [r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        # lbl_B_extalt = [
        #     f'{i} alt.' for i in GRP_FAIR_COMMON] + lbl_B_bin[3:6] + [
        #     r'$\mathbf{df}^\text{avg}$',
        #     r'$\hat{\mathbf{df}}^\text{avg}$']
        lbl_B_ext = tmp_ext + lbl_B_bin[3:6] + [
            r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        lbl_B_extalt = tmp_ext_alt + lbl_B_bin[3:6] + [
            r'$\mathbf{df}^\text{avg}$',
            r'$\hat{\mathbf{df}}^\text{avg}$']
        Mat_B_bin = df[key_B_bin].values.astype(DTY_FLT).T
        Mat_B_ext = df[key_B_nonbin].values.astype(DTY_FLT).T
        Mat_B_extalt = df[key_B_extalt].values.astype(DTY_FLT).T
        kws = {'cmap_name': 'Blues', 'rotate': 65}
        # analogous_confusion_extended(
        #     df[key_A].values.astype(DTY_FLT).T, Mat_B_bin, lbl_A,
        #     lbl_B_bin, f'{figname}_cont1', **kws)
        # analogous_confusion_extended(
        #     df[key_C].values.astype(DTY_FLT).T, Mat_B_bin, lbl_C,
        #     lbl_B_bin, f'{figname}_cont1p', **kws)
        analogous_confusion_extended(
            df[key_C + key_A].values.astype(DTY_FLT).T, Mat_B_bin,
            lbl_C + lbl_A, lbl_B_bin, f'{figname}_cont1p', **kws)
        kws['cmap_name'] = 'Oranges'
        # analogous_confusion_extended(
        #     df[key_A].values.astype(DTY_FLT).T, Mat_B_ext, lbl_A,
        #     lbl_B_ext, f'{figname}_cont2', **kws)
        # analogous_confusion_extended(
        #     df[key_C].values.astype(DTY_FLT).T, Mat_B_ext, lbl_C,
        #     lbl_B_ext, f'{figname}_cont2p', **kws)
        analogous_confusion_extended(
            df[key_C + key_A].values.astype(DTY_FLT).T, Mat_B_ext,
            lbl_C + lbl_A, lbl_B_ext, f'{figname}_cont2p', **kws)
        kws['cmap_name'] = 'RdPu'
        # analogous_confusion_extended(
        #     df[key_A].values.astype(DTY_FLT).T, Mat_B_extalt,
        #     lbl_A, lbl_B_extalt, f'{figname}_cont3', **kws)
        # analogous_confusion_extended(
        #     df[key_C].values.astype(DTY_FLT).T, Mat_B_extalt,
        #     lbl_C, lbl_B_extalt, f'{figname}_cont3p', **kws)
        analogous_confusion_extended(
            df[key_C + key_A].values.astype(DTY_FLT).T, Mat_B_extalt,
            lbl_C + lbl_A, lbl_B_extalt, f'{figname}_cont3p', **kws)
        return

    def draw_extended_idv_tim(self, df, tag_X, tag_Ys, figname):
        tim_grp, tim_grp_nonbin = tag_X
        tim_idv = tag_Ys[0]  # tim_idv, tim_df, tim_df_pl = tag_Ys
        X = df[tim_grp].values.astype(DTY_FLT)
        Ys = [df[i].values.astype(DTY_FLT) / X for i in [
            tim_grp_nonbin, ] + tim_idv]
        antX = r'T_\text{gf (bin-val)}'      # GF,gf
        antY = r'T_\text{if (multival)}'
        annots = [f'${antX}$ (sec)', f'${antY}$',
                  f'${antY}={antX}$']  # antYs[1]
        antZs = [r'$T_\text{gf (multival)}$',
                 # r'$T_{\text{GEI (} \alpha\text{=0.5)}}$',
                 r'$T_\text{GEI}$',    # alpha 0-1 in list
                 r'$T_\text{Theil}$', r'$T_\text{DR}$']
        # Z = df['idv_ptb'].values.astype(DTY_FLT) / X
        # Z = Ys[:-1] + [Z, ]
        annots[2] = r'$T_\text{if}= T_\text{gf (bin-val)}$'

        kws = {'snspec': 'sty7'}  # 'sty6' # {'linreg': True,
        annots[1] = r'$\frac{ T_\text{if (multival)} }{ T_\text{gf (bin-val)} }-1$'
        # multi_lin_reg_without_distr(
        #     X, [i - 1. for i in Ys], antZs, annots,
        #     f'{figname}_tim_st6a', **kws)
        # multi_lin_reg_without_distr(
        #     X, [i - 1. for i in Z], antZs, annots,
        #     f'{figname}_tim_st7a', **kws)
        multi_lin_reg_without_distr(
            X, [i - 1. for i in Ys[:-1]], antZs[:-1], annots,
            f'{figname}_tim_st8a', **kws)
        # multi_lin_reg_without_distr(
        #     X, [i - 1. for i in Ys[1:]], antZs[1:], annots,
        #     f'{figname}_tim_st9a', **kws)
        # annots[1] = r'$\lg(\frac{ T_\text{IF (multival)} }{ T_\text{GF (bin-val)} })$'
        annots[1] = r'$\lg(\frac{ T_\text{if (multival)} }{ T_\text{gf (bin-val)} })$'
        # multi_lin_reg_without_distr(
        #     X, [np.log10(i) for i in Ys], antZs, annots,
        #     f'{figname}_tim_st6b', **kws)
        # multi_lin_reg_without_distr(
        #     X, [np.log10(i) for i in Z], antZs, annots,
        #     f'{figname}_tim_st7b', **kws)
        multi_lin_reg_without_distr(
            X, [np.log10(i) for i in Ys[:-1]], antZs[:-1], annots,
            f'{figname}_tim_st8b', **kws)
        # multi_lin_reg_without_distr(
        #     X, [np.log10(i) for i in Ys[1:]], antZs[1:], annots,
        #     f'{figname}_tim_st9b', **kws)

        # fgn = f'{figname}_df_tim'
        # self.sub_draw_idv_df(df, tim_df, tim_df_pl, tag_X, X, fgn, kws)
        return

    # def sub_draw_idv_bin(self, df, tag_X, tag_Ys, figname):
    #     return
    # def sub_draw_idv_nonbin(self, df, tag_X, tag_Ys, figname):
    #     return

    def sub_draw_idv_df(self, df, tim_df, tim_df_pl, tag_X, X, fgn, kws):
        antZs_drt = [r'$T_\text{gf (multival)}$',
                     r'$T_{\mathbf{df}_\text{prev} \text{ (bin-val)}}$',
                     r'$T_{\mathbf{df} \text{ (multival)}}$',
                     r'$T_{\mathbf{df} \text{ intersectional}}$']
        annots_drt = [r'$T_\text{gf (bin-val)}$ (sec)', '',
                      r'$T_{\mathbf{df}} =T_\text{gf (bin-val)}$']
        Ys = [df[i].values.astype(DTY_FLT) / X for i in tag_X[
            1:] + tim_df[:2] + tim_df_pl[:1]]
        annots_drt[1] = r'$\frac{ T_{\mathbf{df} \text{ (multival)}} }{T_\text{gf (bin-val)}}-1$'
        multi_lin_reg_without_distr(
            X, [i - 1. for i in Ys], antZs_drt, annots_drt,
            f'{fgn}_da', **kws)
        # multi_lin_reg_without_distr(
        #     X, [i - 1. for i in Ys[1:]], antZs_drt[1:],
        #     annots_drt, f'{fgn}_d1p', **kws)
        annots_drt[1] = r'$\lg(\frac{ T_{\mathbf{df} \text{ (multival)}} }{T_\text{gf (bin-val)}})$'
        multi_lin_reg_without_distr(
            X, [np.log10(i) for i in Ys], antZs_drt, annots_drt,
            f'{fgn}_db', **kws)
        # multi_lin_reg_without_distr(
        #     X, [np.log10(i) for i in Ys[1:]], antZs_drt[1:],
        #     annots_drt, f'{fgn}_d2p', **kws)

        antZs_app = [
            r'$T_\text{gf (multival)}$',
            r'$T_{\hat{\mathbf{df}}_\text{prev} \text{ (bin-val)}}$',
            r'$T_{\hat{\mathbf{df}} \text{ (multival)}}$',
            r'$T_{\hat{\mathbf{df}} \text{ intersectional}}$']
        annots_app = [r'$T_\text{gf (bin-val)}$ (sec)', '',
                      r'$T_{\hat{\mathbf{df}}} =T_\text{gf (bin-val)}$']
        Ys = [df[i].values.astype(DTY_FLT) / X for i in tag_X[
            1:] + tim_df[2:] + tim_df_pl[1:]]
        annots_app[1] = r'$\frac{ T_{\hat{\mathbf{df}} \text{ (multival)}} }{T_\text{gf (bin-val)}}-1$'
        multi_lin_reg_without_distr(
            X, [i - 1. for i in Ys], antZs_app, annots_app,
            f'{fgn}_aa', **kws)
        # multi_lin_reg_without_distr(
        #     X, [i - 1. for i in Ys[1:]], antZs_app[1:],
        #     annots_app, f'{fgn}_a1p', **kws)
        annots_app[1] = r'$\lg(\frac{ T_{\hat{\mathbf{df}} \text{ (multival)}} }{T_\text{gf (bin-val)}})$'
        multi_lin_reg_without_distr(
            X, [np.log10(i) for i in Ys], antZs_app, annots_app,
            f'{fgn}_ab', **kws)
        # multi_lin_reg_without_distr(
        #     X, [np.log10(i) for i in Ys[1:]], antZs_app[1:],
        #     annots_app, f'{fgn}_a2p', **kws)
        return


class PlotA_fair_ens(PlotA_initial):
    def __init__(self):
        """PlotA_fair_ens
        """
        pass

    def schedule_mspaint(self, raw_dframe, figname=''):
        _, id_set = self.recap_sub_data(raw_dframe, sa_ir=3, sa_r=4)
        mk = 'tst'  # flag,mark
        first_incl = verbose = False
        # df_bin = self.obtain_binval_senatt(raw_dframe, id_set, mk)
        df_nonbin = self.obtain_multival_senatt(raw_dframe, id_set, mk,
                                                first_incl=first_incl)
        tag_acc, tag_sa1, _ = self.obtain_tag_col(mk)
        tag_acc = tag_acc[: -2]

        tmp = tag_sa1[-6: -3]
        df_nonbin['extGrp'] = df_nonbin[
            tmp[0]] + df_nonbin[tmp[1]] + df_nonbin[tmp[2]]
        tmp = tag_sa1[-3:]
        df_nonbin['extAlt'] = df_nonbin[tmp[0]] + df_nonbin[
            tmp[1]] + df_nonbin[tmp[2]] + df_nonbin['extGrp']
        pick = [0, 4, 5]  # 1,2,3,] # ,6,7]
        col_grp = tag_sa1[:3] + [tag_sa1[16 + 3], tag_sa1[27 + 3]]
        col_ext = tag_sa1[4:10][:3] + [tag_sa1[16 + 7], tag_sa1[27 + 7]]
        col_ext_alt = tag_sa1[10:16][:3] + [
            tag_sa1[16 + 10], tag_sa1[27 + 10]]
        self.draw_trade_off(df_nonbin, pick, tag_acc[:16] + [
            tag_acc[19 + 2], tag_acc[19 + 4], tag_acc[15 + 3], ], [
            col_grp, col_ext, col_ext_alt], f'{figname}_to')
        self.draw_extended_grp_scat(
            df_nonbin, col_grp, col_ext, col_ext_alt,
            f'{figname}_scat', verbose)

        tim_idv = [tag_acc[16 + 3], ] + tag_acc[16 + 4 + 4:][:2]
        tim_idv = tim_idv[:: -1]   # DR,Theil,GEI: then reverse
        tim_df_pl = [tag_acc[26:][6], tag_acc[26:][6 + 7],
                     ]  # df/hat_df multiver (df intersectional)
        tim_grp = [tag_sa1[3], tag_sa1[16], 'extGrp', 'extAlt']  # three
        tim_df = [tag_sa1[20], tag_sa1[27], tag_sa1[27 + 4], tag_sa1[
            27 + 4 + 7]]  # df4one sen-att: bin-val, multival, hat_df x2
        df_nonbin['idv_ptb'] = df_nonbin[tag_acc[-2]] + df_nonbin[
            tim_idv[-1]]  # 'idvDR_perturb','idv_dr_ptb', 'idvDR_'
        self.draw_extended_idv_tim(df_nonbin, tim_grp[:2], [
            tim_idv, tim_df, tim_df_pl], figname + '_idv')

        # self.draw_extended_grp(df_nonbin, tag_sa1[3], [tag_sa1[16]], (
        #     # 'Group fairness (bin-val)',
        #     # 'Extended group fairness (multival)'))
        #     'Grp (bin-val)', 'Ext.grp (multival)'))
        # self.draw_extended_grp(df_nonbin, tag_sa1[3], [tag_sa1[16]],
        #                        'Grp (bin-val)', ['Ext.grp (multival)'],
        #                        figname)
        self.draw_extended_grp_tim(df_nonbin, tag_sa1[3], [
            tag_sa1[16], 'extGrp', 'extAlt'], figname + '_grp')
        self.draw_extended_hfm_tim(df_nonbin, tag_sa1[20], [
            tag_sa1[27], tag_sa1[27 + 4], tag_sa1[27 + 4 + 7]],
            figname + '_hfm')
        # # self.draw_extended_hfm_tim(df_nonbin, tag_sa1[20], [
        # #     tag_sa1[27], ], figname + '_hfm_prim')
        # self.draw_extended_grp_scat(df_nonbin, tag_sa1[:3] + [
        #     # tag_sa1[19]],  # tag_sa1[4:4 + 6], tag_sa1[4 + 6: 4 + 12],
        #     # tag_sa1[4:10] + [tag_sa1[19 + 4], tag_sa1[19 + 7]],
        #     # tag_sa1[10:16] + [tag_sa1[26 + 3], tag_sa1[26 + 7]],
        #     tag_sa1[16 + 3], tag_sa1[27 + 3]],
        #     tag_sa1[4:10][:3] + [tag_sa1[16 + 7], tag_sa1[27 + 7]],
        #     tag_sa1[10:16][:3] + [tag_sa1[16 + 10], tag_sa1[27 + 10]],
        #     figname + '_scat', verbose)

        # # self.obtain_sing_dat_cls(0, 2, raw_dframe, id_set, mk)
        # # self.depict_separately(0, 2, mk)
        # self.depict_separately(0, 2, raw_dframe, id_set, mk,
        #                        figname + '_radar')
        # for pks in [2, 3, 4]:
        #     self.depict_separately(pks, 2, raw_dframe, id_set,
        #                            mk, figname + '_radar')
        fgn = f'{figname}_radar'
        for pkc in [0, 1, 2, 6, 10]:  # range(3+4+4):
            for pks in [2, 3, 4]:
                self.depict_separately(
                    pks, pkc, raw_dframe, id_set, mk, fgn)
        if not first_incl:
            return
        for pkc in [0, 1, 2, 6]:     # range(3+4):
            self.depict_separately(  # pks = 0
                0, pkc, raw_dframe, id_set, mk, fgn)
        return

    def schedule_mspaint_avg(self, raw_dframe, figname=''):
        _, id_set = self.recap_sub_data(raw_dframe, sa_ir=3, sa_r=4)
        mk, first_incl, verbose = 'tst', False, False
        df_nonbin = self.obtain_multival_senatt(
            raw_dframe, id_set, mk, first_incl=first_incl)
        tag_acc, tag_sa1, _ = self.obtain_tag_col(mk)
        tag_acc = tag_acc[: -2]
        tmp = tag_sa1[-6: -3]
        df_nonbin['extGrp'] = df_nonbin[tmp[
            0]] + df_nonbin[tmp[1]] + df_nonbin[tmp[2]]    # TimeCost
        tmp = tag_sa1[-3:]
        df_nonbin['extAlt'] = df_nonbin[tmp[0]] + df_nonbin[tmp[
            1]] + df_nonbin[tmp[2]] + df_nonbin['extGrp']  # TimeCost
        pick = [0, 4, 5]

        # extension in average forms, above is maximal forms
        col_grp = tag_sa1[:3] + [tag_sa1[16 + 3], tag_sa1[27 + 3]]
        col_ext = tag_sa1[4:10][-3:] + [tag_sa1[23], tag_sa1[34]]
        col_ext_alt = tag_sa1[10:16][-3:] + [tag_sa1[26], tag_sa1[37]]
        self.avg_draw_trade_off(df_nonbin, pick, tag_acc[
            :16] + [tag_acc[21], tag_acc[23], tag_acc[18], ], [
            col_grp, col_ext, col_ext_alt],
            f'{figname}_to_avg')  # f'{figname}_avg_to')
        self.avg_draw_extended_grp_scat(
            df_nonbin, col_grp, col_ext + tag_sa1[4:7],
            col_ext_alt + tag_sa1[10:13],
            # f'{figname}_avg_scat', verbose)
            f'{figname}_scat_avg', verbose)
        fgn = f'{figname}_radar_avg'  # f'{figname}_avg_radar'
        for pkc in [0, 1, 2, 6, 10]:
            for pks in [2, 3, 4]:
                self.avg_depict_separately(
                    pks, pkc, raw_dframe, id_set, mk, fgn)
                if pkc == 2:
                    continue
                os.remove(f'{fgn[:-4]}_s{pks}c{pkc}_ori.pdf')
        # if not first_incl:
        #     return
        # for pkc in [0, 1, 2, 6]:
        #     self.avg_depict_separately(
        #         0, pkc, raw_dframe, id_set, mk, fgn)

        col_ext_max = tag_sa1[4:10][:3] + [tag_sa1[23], tag_sa1[34]]
        col_ext_alt_max = tag_sa1[10:16][:3] + [
            tag_sa1[26], tag_sa1[37]]
        self.avg_draw_trade_off_alt(
            df_nonbin, pick, tag_acc[:16] + [
                tag_acc[21], tag_acc[23], tag_acc[18], ], [
                col_grp, col_ext_max, col_ext_alt_max,
                col_ext, col_ext_alt], f'{figname}_to_alt')
        self.avg_draw_incompatible_alt(
            df_nonbin, tag_acc[:16] + [
                tag_acc[21], tag_acc[23], tag_acc[18], ], [
                col_grp, col_ext_max, col_ext_alt_max,
                col_ext, col_ext_alt], f'{figname}_nc')
        return

    def avg_depict_separately(self, pick_set, pick_clf, df, id_set,
                              tag_mk='tst', fgn='', multival=True):
        #                       verbose=True):
        # if not verbose:
        #     os.remove(f'{fgn[:-4]}_s{pick_set}c{pick_clf}_ori.pdf')

        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(tag_mk)
        tag_acc = tag_acc[: -2]
        df_alt = self.obtain_sing_dat_cls(
            pick_set, pick_clf, tag_acc, tag_sa1, tag_sa2,
            df, id_set, multival)
        sub_grp = tag_sa1[:3] + [tag_sa1[16 + 3], tag_sa1[27 + 3]]
        sub_ext = tag_sa1[4:10][:3] + [tag_sa1[16 + 7], tag_sa1[27 + 7]]
        sub_ext_alt = tag_sa1[10:16][:3] + [tag_sa1[
            16 + 10], tag_sa1[27 + 10]]
        sub_idv = tag_acc[16:16 + 4 + 6]  # dr 4+ GEI.alph 3+ Theil+Tx2
        sub_idv = [sub_idv[2], ] + sub_idv[4:-2]
        sub_idv = [sub_idv[0], sub_idv[2], sub_idv[-1]]

        sub_ext_avg = tag_sa1[4:10][-3:] + [tag_sa1[23], tag_sa1[34]]
        sub_ext_alt_avg = tag_sa1[10:16][-3:] + [
            tag_sa1[26], tag_sa1[37]]
        currX = sub_grp[:3] + sub_idv + sub_grp[-2:]
        labels = GRP_FAIR_COMMON + [
            'DR', r'GEI ($\alpha$=0.5)', 'Theil',
            r'$\mathbf{df}_\text{prev}$',
            r'$\hat{\mathbf{df}}_\text{prev}$']
        df_tmp = df_alt[currX]
        for i in currX:
            df_tmp.loc[:, i] = float(df_tmp[i].mean())
        # radar_chart(df_tmp, currX, annotX=labels,
        #             figname=f'{fgn}_s{pick_set}c{pick_clf}_ori',
        #             clockwise=True)

        df_tmp = df_tmp.reset_index(drop=True)
        df_tmp_tmp = df_alt[sub_ext]
        for i, j in zip(sub_grp, sub_ext):
            df_tmp.loc[1, i] = float(df_tmp_tmp[j].mean())
        df_tmp_tmp = df_alt[sub_ext_alt]
        for i, j in zip(sub_grp, sub_ext_alt):
            df_tmp.loc[2, i] = float(df_tmp_tmp[j].mean())

        df_tmp_tmp = df_alt[sub_ext_avg]
        for i, j in zip(sub_grp, sub_ext_avg):
            df_tmp.loc[3, i] = float(df_tmp_tmp[j].mean())
        df_tmp_tmp = df_alt[sub_ext_alt_avg]
        for i, j in zip(sub_grp, sub_ext_alt_avg):
            df_tmp.loc[4, i] = float(df_tmp_tmp[j].mean())
        annotY = ['ori', 'ext', 'alt', 'ext (avg)', 'alt (avg)']
        # pdb.set_trace()
        radar_chart(df_tmp, currX, labels, annotY,
                    figname=f'{fgn}_s{pick_set}c{pick_clf}')
        return

    def avg_draw_extended_grp_scat(self, df, tag_grp, tag_ext,
                                   tag_ext_alt, figname,
                                   verbose=False, ver_mark=' (avg)'):
        labels = ['ori', 'ext', 'alt',
                  f'ext{ver_mark}', f'alt{ver_mark}']
        lbl_dim2 = GRP_FAIR_COMMON + [
            r'$\mathbf{df}_\text{prev}$',
            r'$\hat{\mathbf{df}}_\text{prev}$']
        fgn = figname.replace('_avg', '')
        multi_boxplot_rect(df, tag_grp[:3], tag_ext[:3],
                           labels =['ori'] + labels[-2:],
                           figname=f'{fgn}_grpext_avg',
                           annotX=lbl_dim2[:3], locate="upper left")
        multi_boxplot_rect(
            df, tag_grp[:3], tag_ext[:3], tag_ext_alt[:3],
            labels =['ori'] + labels[-2:],
            figname=f'{fgn}_grpalt_avg',
            annotX=lbl_dim2[:3], locate="upper left")
        multi_boxplot_rect(
            df, tag_grp[:3], tag_ext[-3:], tag_ext_alt[-3:],
            tag_ext[:3], tag_ext_alt[:3],
            figname=f'{fgn}_group_max_avg',
            annotX=lbl_dim2[:3], locate="upper left",
            figsize='M-NT')

        multi_boxplot_rect(
            df, tag_grp[:3], tag_ext[-3:], labels =labels[:3],
            # figname='{}_grpext'.format(figname.replace('avg', 'max')),
            figname=f'{fgn}_grpext',  # f'{fgn}_grpext_max',
            annotX=lbl_dim2[:3], locate="upper left")
        multi_boxplot_rect(
            df, tag_grp[:3], tag_ext[-3:], tag_ext_alt[-3:],
            labels =labels[:3],
            # figname=f'{figname.replace('avg', 'max')}_grpalt',
            # figname='{}_grpext'.format(figname.replace('avg', 'max')),
            figname=f'{fgn}_grpalt',  # f'{fgn}_grpalt_max',
            annotX=lbl_dim2[:3], locate="upper left")
        # os.remove(f'{figname[:-9]}_scat_grpalt.pdf')
        # os.remove(f'{figname[:-9]}_scat_grpext.pdf')

        # labels = labels[:3] if not ver_mark else ['ori'] + labels[-2:]
        # lbl_hfm = [[r'$\mathbf{df}_\text{prev}$', r'$\mathbf{df}$',
        #             r'$\mathbf{df}^{avg}$'], [
        #     r'$\hat{\mathbf{df}}_\text{prev}$', r'$\hat{\mathbf{df}}$',
        #     r'$\hat{\mathbf{df}}^{avg}$'], ]
        # for i, tg in enumerate(tag_grp):
        #     data = [df[tg].values.astype(DTY_FLT),
        #             df[tag_ext[i]].values.astype(DTY_FLT),
        #             df[tag_ext_alt[i]].values.astype(DTY_FLT)]
        #     fgn = '{}_{}'.format(
        #         figname, f'grp{i+1}' if i < 3 else f'hfm{i+3}')
        #     multi_boxplot_rect(df, [tg, tag_ext[
        #         i], tag_ext_alt[i]], figname=fgn,
        #         annotX=labels if i < 3 else lbl_hfm[i - 3])  # not tag_Xs
        # multi_boxplot_rect(df, tag_grp, tag_ext[:5],
        #                    figname=f'{figname}_dim2', annotX=lbl_dim2)
        # multi_boxplot_rect(df, tag_grp, tag_ext[:5], tag_ext_alt[:5],
        #                    figname=f'{figname}_dim3', annotX=lbl_dim2)
        return

    def avg_draw_trade_off(self, df, pick, tag_X, tag_Ys, figname,
                           ver_mark=' (avg)'):
        annotZs = GRP_FAIR_COMMON + [
            r'GEI ($\alpha$=0.5)', 'Theil', 'DR']
        annotY = 'Extended group fairness (multival)'
        # tmp_ext = ['{:6s} ext{}'.format(
        #     i, ver_mark) for i in annotZs[:3]]
        # tmp_ext_alt = ['{:6s} alt{}'.format(  # 'ext. alt{}'
        #     i, ver_mark) for i in annotZs[:3]]
        # tmp_ext[1] = f'{annotZs[1]} ext{ver_mark}'
        # tmp_ext_alt[1] = f'{annotZs[1]} alt{ver_mark}'
        tmp_ext = [r'$\text{DP}^\text{ext(avg)}$',
                   r'$\text{EOpp}^\text{ext(avg)}$',
                   r'$\text{PP}^\text{ext(avg)}$', ]
        tmp_ext_alt = [r'$\text{DP}^\text{alt(avg)}$',
                       r'$\text{EOpp}^\text{alt(avg)}$',
                       r'$\text{PP}^\text{alt(avg)}$', ]

        for pk in pick:
            annotX = self._perf_metric[pk]
            line_reg_with_marginal_distr(
                df, tag_X[pk], 'Fairness', tag_Ys[1][:3],
                # [f'{i} ext.{ver_mark}' for i in annotZs[:3]],
                tmp_ext, annotX=annotX, annotY=annotY,
                snspec='sty4b', figname=f'{figname}_to{pk}_s2')
            line_reg_with_marginal_distr(
                # df, tag_X[pk], 'Fairness', tag_Ys[0][:3],
                # [f'{i} ext. alt{ver_mark}' for i in annotZs[:3]],
                df, tag_X[pk], 'Fairness', tag_Ys[2][:3],
                tmp_ext_alt, annotX=annotX, annotY=annotY.replace(
                    'Extended', 'Alternative extended'),
                snspec='sty4b', figname=f'{figname}_to{pk}_s3')

        key_A = [tag_X[:8][i] for i in pick]
        key_C = [tag_X[8:16][i] for i in pick]
        # key_B_bin = tag_Ys[0][:3] + tag_X[-3:] + tag_Ys[0][-2:]
        key_B_nonbin = tag_Ys[1][:3] + tag_X[-3:] + tag_Ys[1][-2:]
        key_B_extalt = tag_Ys[2][:3] + tag_X[-3:] + tag_Ys[2][-2:]
        lbl_A = [self._perf_metric[i] for i in pick]
        lbl_C = [self._dal_metric[i] for i in pick]
        lbl_B_bin = annotZs + [r'$\mathbf{df}_\text{prev}$',
                               r'$\hat{\mathbf{df}}_\text{prev}$']
        # Mat_B_bin = df[key_B_bin].values.astype(DTY_FLT).T
        Mat_B_ext = df[key_B_nonbin].values.astype(DTY_FLT).T
        Mat_B_extalt = df[key_B_extalt].values.astype(DTY_FLT).T
        kws = {'cmap_name': 'Blues', 'rotate': 65}

        # lbl_B_ext = [f'{i} ext{ver_mark}' for i in GRP_FAIR_COMMON
        #              ] + lbl_B_bin[3:6] + [
        #     r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        # lbl_B_extalt = [f'{i} alt{ver_mark}' for i in GRP_FAIR_COMMON
        #                 ] + lbl_B_bin[3:6] + [
        #     r'$\mathbf{df}^\text{avg}$',
        #     r'$\hat{\mathbf{df}}^\text{avg}$']
        lbl_B_ext = tmp_ext + lbl_B_bin[3:6] + [
            r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        lbl_B_extalt = tmp_ext_alt + lbl_B_bin[3:6] + [
            r'$\mathbf{df}^\text{avg}$',
            r'$\hat{\mathbf{df}}^\text{avg}$']
        kws['cmap_name'] = 'Oranges'
        analogous_confusion_extended(
            df[key_C + key_A].values.astype(DTY_FLT).T, Mat_B_ext,
            lbl_C + lbl_A, lbl_B_ext, f'{figname}_cont2p', **kws)
        kws['cmap_name'] = 'RdPu'
        analogous_confusion_extended(
            df[key_C + key_A].values.astype(DTY_FLT).T, Mat_B_extalt,
            lbl_C + lbl_A, lbl_B_extalt, f'{figname}_cont3p', **kws)
        return

    def avg_draw_trade_off_alt(self, df, pick, tag_X, tag_Ys,
                               figname):
        annotZs = GRP_FAIR_COMMON + [
            r'GEI ($\alpha$=0.5)', 'Theil', 'DR']
        key_A = [tag_X[:8][i] for i in pick]
        key_C = [tag_X[8:16][i] for i in pick]
        key_B_bin = tag_Ys[0][:3] + tag_X[-3:] + tag_Ys[0][-2:]
        lbl_A = [self._perf_metric[i] for i in pick]
        lbl_C = [self._dal_metric[i] for i in pick]
        lbl_B_bin = annotZs + [r'$\mathbf{df}_\text{prev}$',
                               r'$\hat{\mathbf{df}}_\text{prev}$']
        Mat_B_bin = df[key_B_bin].values.astype(DTY_FLT).T
        kws = {'cmap_name': 'Blues', 'rotate': 65}

        key_B_nonbin = tag_Ys[1][:3] + tag_Ys[3][:3] + tag_Ys[1][-2:]
        key_B_extalt = tag_Ys[2][:3] + tag_Ys[4][:3] + tag_Ys[2][-2:]
        # lbl_B_ext = [f'{i} ext' for i in GRP_FAIR_COMMON] + [
        #     f'{i} ext(avg)' for i in GRP_FAIR_COMMON] + [
        #     r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        # lbl_B_extalt = [f'{i} alt' for i in GRP_FAIR_COMMON] + [
        #     f'{i} alt(avg)' for i in GRP_FAIR_COMMON] + [
        #     r'$\mathbf{df}^\text{avg}$',
        #     r'$\hat{\mathbf{df}}^\text{avg}$']
        Mat_B_ext = df[key_B_nonbin].values.astype(DTY_FLT).T
        Mat_B_extalt = df[key_B_extalt].values.astype(DTY_FLT).T

        lbl_B_ext = [r'$\text{DP}^\text{ext}$',
                     r'$\text{EOpp}^\text{ext}$',
                     r'$\text{PP}^\text{ext}$',
                     r'$\text{DP}^\text{ext(avg)}$',
                     r'$\text{EOpp}^\text{ext(avg)}$',
                     r'$\text{PP}^\text{ext(avg)}$', ] + [
            r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        lbl_B_extalt = [r'$\text{DP}^\text{alt}$',
                        r'$\text{EOpp}^\text{alt}$',
                        r'$\text{PP}^\text{alt}$',
                        r'$\text{DP}^\text{alt(avg)}$',
                        r'$\text{EOpp}^\text{alt(avg)}$',
                        r'$\text{PP}^\text{alt(avg)}$', ] + [
            r'$\mathbf{df}^\text{avg}$',
            r'$\hat{\mathbf{df}}^\text{avg}$']

        fgn = figname[:-4]  # figname.replace('_alt', '')
        os.remove(f'{fgn}_cont1p.pdf')
        os.remove(f'{fgn}_cont2p.pdf')
        os.remove(f'{fgn}_cont3p.pdf')
        os.remove(f'{fgn}_avg_cont2p.pdf')
        os.remove(f'{fgn}_avg_cont3p.pdf')
        Mat_C_A = df[key_C + key_A].values.astype(DTY_FLT).T
        analogous_confusion_extended(
            Mat_C_A, Mat_B_bin, lbl_C + lbl_A, lbl_B_bin,
            f'{fgn}_cont1p', **kws)

        kws['cmap_name'] = 'Oranges'
        analogous_confusion_extended(
            Mat_C_A, Mat_B_ext, lbl_C + lbl_A, lbl_B_ext,
            f'{figname}_cont2p', **kws)
        kws['cmap_name'] = 'RdPu'
        analogous_confusion_extended(
            Mat_C_A, Mat_B_extalt, lbl_C + lbl_A,
            lbl_B_extalt, f'{figname}_cont3p', **kws)
        return

    def avg_draw_incompatible_alt(self, df, tag_X, tag_Ys,
                                  figname, verbose=False):
        annotZs = GRP_FAIR_COMMON + [
            r'GEI ($\alpha$=0.5)', 'Theil', 'DR']
        # ta, ra = f"{'':<5}", f"{'':>5}"
        # tmp_ext = [r'$\text{DP}^\text{ext}$' + ta,
        #            r'$\text{EOpp}^\text{ext}$',
        #            r'$\text{PP}^\text{ext}$' + ta, ]
        # tmp_ext_alt = [r'$\text{DP}^\text{alt}$' + ta,
        #                r'$\text{EOpp}^\text{alt}$',
        #                r'$\text{PP}^\text{alt}$' + ta, ]
        # tmp_ext_avg = [r'$\text{DP}^\text{ext(avg)}$' + ra,
        #                r'$\text{EOpp}^\text{ext(avg)}$',
        #                r'$\text{PP}^\text{ext(avg)}$' + ra]
        # tmp_ext_alt_avg = [r'$\text{DP}^\text{alt(avg)}$' + ra,
        #                    r'$\text{EOpp}^\text{alt(avg)}$',
        #                    r'$\text{PP}^\text{alt(avg)}$' + ra]
        # annotZs[0] += f"{'':>4}"  # ta
        # annotZs[2] += f"{'':<4}"  # ra
        #
        # kws = {'annotY': 'Group fairness', 'snspec': 'sty5b'}  # invt_a=True
        # for i, pk in enumerate(tag_X[-3:]):
        #     annotX = annotZs[i + 3]
        #     annotX = f'Individual fairness: {annotX}'
        #     # annotX = f'Individual fairness ({annotX})'
        #     # f'{fgn}_bin' _ext,_alt,_ext_avg,_alt_avg
        #     fgn = f'{figname}_corr{i+3}'
        #     line_reg_with_marginal_distr(
        #         df, pk, 'Fairness', tag_Ys[0][:3], annotZs[:3],
        #         annotX=annotX, figname=f'{fgn}_g1', **kws)
        #     line_reg_with_marginal_distr(
        #         df, pk, 'Fairness', tag_Ys[1][:3], tmp_ext,
        #         annotX=annotX, figname=f'{fgn}_g2', **kws)
        #     line_reg_with_marginal_distr(
        #         df, pk, 'Fairness', tag_Ys[2][:3], tmp_ext_alt,
        #         annotX=annotX, figname=f'{fgn}_g3', **kws)
        #     line_reg_with_marginal_distr(
        #         df, pk, 'Fairness', tag_Ys[3][:3], tmp_ext_avg,
        #         annotX=annotX, figname=f'{fgn}_g4', **kws)
        #     line_reg_with_marginal_distr(
        #         df, pk, 'Fairness', tag_Ys[4][:3], tmp_ext_alt_avg,
        #         annotX=annotX, figname=f'{fgn}_g5', **kws)

        ta, ra = f"{'':<7}", f"{'':>8}"
        tmp_ext = [r'$\text{DP}^\text{ext}$' + ta,
                   r'$\text{EOpp}^\text{ext}$' + ta,
                   r'$\text{PP}^\text{ext}$' + ta, ]
        tmp_ext_alt = [r'$\text{DP}^\text{alt}$' + ra,
                       r'$\text{EOpp}^\text{alt}$' + ra,
                       r'$\text{PP}^\text{alt}$' + ra, ]
        tmp_ext_avg = [r'$\text{DP}^\text{ext(avg)}$',
                       r'$\text{EOpp}^\text{ext(avg)}$',
                       r'$\text{PP}^\text{ext(avg)}$']
        tmp_ext_alt_avg = [r'$\text{DP}^\text{alt(avg)}$' + ' ',
                           r'$\text{EOpp}^\text{alt(avg)}$' + ' ',
                           r'$\text{PP}^\text{alt(avg)}$' + ' ']
        annotZs[0] += ra + f"{'':>3}"
        annotZs[2] += ra + f"{'':>2}"
        annotZs[1] += ra + f"{'':<3}"
        grp1 = [y[0] for y in tag_Ys]
        grp2 = [y[1] for y in tag_Ys]
        grp3 = [y[2] for y in tag_Ys]
        lbl_g1 = [annotZs[0], tmp_ext[0], tmp_ext_alt[0],
                  tmp_ext_avg[0], tmp_ext_alt_avg[0]]
        lbl_g2 = [annotZs[1], tmp_ext[1], tmp_ext_alt[1],
                  tmp_ext_avg[1], tmp_ext_alt_avg[1]]
        lbl_g3 = [annotZs[2], tmp_ext[2], tmp_ext_alt[2],
                  tmp_ext_avg[2], tmp_ext_alt_avg[2]]
        kws = {'snspec': 'sty5b'}  # kws.pop('annotY')
        for i, pk in enumerate(tag_X[-3:]):
            annotX = annotZs[i + 3]
            fgn = f'{figname}_corr{i+4}'
            line_reg_with_marginal_distr(
                df, pk, 'Fairness', grp1, lbl_g1, annotX=annotX,
                annotY=lbl_g1[0], figname=f'{fgn}_grp1', **kws)
            line_reg_with_marginal_distr(
                df, pk, 'Fairness', grp2, lbl_g2, annotX=annotX,
                annotY=lbl_g2[0], figname=f'{fgn}_grp2', **kws)
            line_reg_with_marginal_distr(
                df, pk, 'Fairness', grp3, lbl_g3, annotX=annotX,
                annotY=lbl_g3[0], figname=f'{fgn}_grp3', **kws)

        hfm_drt = [tag_Ys[0][-2], tag_Ys[1][-2], tag_Ys[2][-2]]
        hfm_app = [tag_Ys[0][-1], tag_Ys[1][-1], tag_Ys[2][-1]]
        lbl_drt = [r'$\mathbf{df}_\text{prev}$', r'$\mathbf{df}$',
                   r'$\mathbf{df}^\text{avg}$']
        lbl_app = [r'$\hat{\mathbf{df}}_\text{prev}$',
                   r'$\hat{\mathbf{df}}$',
                   r'$\hat{\mathbf{df}}^\text{avg}$']
        for k, (pi, pj) in enumerate(zip(hfm_drt, hfm_app)):
            fgn = f'{figname}_df{k}'
            # _drt_g1,_drt_g2,_drt_g3,_app_g1,_app_g2,_app_g3
            line_reg_with_marginal_distr(
                df, pi, 'Fairness', grp1, lbl_g1, annotX=lbl_drt[k],
                annotY=lbl_g1[0], figname=f'{fgn}_g1d', **kws)
            line_reg_with_marginal_distr(
                df, pi, 'Fairness', grp2, lbl_g2, annotX=lbl_drt[k],
                annotY=lbl_g2[0], figname=f'{fgn}_g2d', **kws)
            line_reg_with_marginal_distr(
                df, pi, 'Fairness', grp3, lbl_g3, annotX=lbl_drt[k],
                annotY=lbl_g3[0], figname=f'{fgn}_g3d', **kws)
            if not verbose:
                continue
            line_reg_with_marginal_distr(
                df, pj, 'Fairness', grp1, lbl_g1, annotX=lbl_app[k],
                annotY=lbl_g1[0], figname=f'{fgn}_g1a', **kws)
            line_reg_with_marginal_distr(
                df, pj, 'Fairness', grp2, lbl_g2, annotX=lbl_app[k],
                annotY=lbl_g2[0], figname=f'{fgn}_g2a', **kws)
            line_reg_with_marginal_distr(
                df, pj, 'Fairness', grp3, lbl_g3, annotX=lbl_app[k],
                annotY=lbl_g3[0], figname=f'{fgn}_g3a', **kws)
        return


class PlotA_norm_cls(PlotA_initial):
    def __init__(self):
        pass

    def schedule_mspaint(self, raw_dframe, figname=''):
        # nb_set, id_set = self.recap_sub_data(
        #     raw_dframe, sa_ir=11, sa_r=0)
        _, id_set = self.recap_sub_data(raw_dframe, sa_ir=11, sa_r=0)
        mk = 'tst'
        first_incl = verbose = False
        df_nonbin = self.obtain_multival_senatt(
            raw_dframe, id_set, mk, first_incl)
        tag_acc, tag_sa1, _ = self.obtain_tag_col(mk)
        tag_acc = tag_acc[: -2]

        tmp = tag_sa1[-6: -3]
        df_nonbin['extGrp'] = df_nonbin[
            tmp[0]] + df_nonbin[tmp[1]] + df_nonbin[tmp[2]]
        tmp = tag_sa1[-3:]
        df_nonbin['extAlt'] = df_nonbin[tmp[0]] + df_nonbin[
            tmp[1]] + df_nonbin[tmp[2]] + df_nonbin['extGrp']
        self.draw_extended_grp_tim(df_nonbin, tag_sa1[
            3], [tag_sa1[16], 'extGrp', 'extAlt'],
            f'{figname}_grp', verbose)
        self.draw_extended_hfm_tim(df_nonbin, tag_sa1[20], [
            tag_sa1[27], tag_sa1[27 + 4], tag_sa1[27 + 4 + 7]],
            f'{figname}_hfm')
        self.draw_extended_grp_scat(df_nonbin, tag_sa1[:3] + [
            tag_sa1[16 + 3], tag_sa1[27 + 3]],
            tag_sa1[4:10][:3] + [tag_sa1[16 + 7], tag_sa1[27 + 7]],
            tag_sa1[10:16][:3] + [tag_sa1[16 + 10], tag_sa1[27 + 10]],
            f'{figname}_scat', verbose)
        fgn = f'{figname}_radar'
        for pks in [2, 3, 4]:
            self.depict_separately(pks, 2, raw_dframe, id_set, mk, fgn)
        if not verbose:
            return
        for pkc in [0, 1, 6, 10]:
            for pks in [2, 3, 4]:
                self.depict_separately(
                    pks, pkc, raw_dframe, id_set, mk, fgn)
        return


# -----------------------------
# Revision

# GRP_FAIR_COMMON[0] = r'$\Delta$' + GRP_FAIR_COMMON[0]
# GRP_FAIR_COMMON[1] = r'$\Delta$' + GRP_FAIR_COMMON[1]
# GRP_FAIR_COMMON[2] = r'$\Delta$' + GRP_FAIR_COMMON[2]

# GRP_FAIR_COMMON = [r'$\Delta$' + i for i in GRP_FAIR_COMMON]
GRP_EXTENSIONS = ['Original', 'Extended', 'Alternative',
                  'Extended (avg.)', 'Alternative (avg.)']


class PlotB_fair_ens(PlotA_fair_ens):
    def draw_extended_grp_scat(self, df, tag_grp, tag_ext, tag_ext_alt,
                               figname, verbose=False):
        labels = ['ori', 'ext', 'alt']
        lbl_hfm = [[r'$\mathbf{df}_\text{prev}$', r'$\mathbf{df}$',
                    r'$\mathbf{df}^{avg}$'], [
            r'$\hat{\mathbf{df}}_\text{prev}$', r'$\hat{\mathbf{df}}$',
            r'$\hat{\mathbf{df}}^{avg}$'], ]
        lbl_dim2 = ['DP', 'EO', 'PQP', r'$\mathbf{df}_\text{prev}$',
                    r'$\hat{\mathbf{df}}_\text{prev}$']
        lbl_dim2[2] = 'PP'
        lbl_dim2[:3] = GRP_FAIR_COMMON

        # multi_boxplot_rect_revised(
        #     # df, tag_grp[:3], None,
        #     df, tag_grp[:3], tag_ext[:3],
        #     figname=f'{figname}_grpext', annotX=lbl_dim2[:3],
        #     locate="upper left")
        multi_boxplot_rect_revised(
            df, tag_grp[:3], tag_ext[:3], tag_ext_alt[:3],
            figname=f'{figname}_grpalt', annotX=lbl_dim2[:3],
            locate="upper left")
        pdb.set_trace()
        if not verbose:
            return

        for i, tg in enumerate(tag_grp):
            fgn = '{}_{}'.format(
                figname, f'grp{i+1}' if i < 3 else f'hfm{i+3}')
            multi_boxplot_rect(df, [tg, tag_ext[
                i], tag_ext_alt[i]], figname=fgn,
                annotX=labels if i < 3 else lbl_hfm[i - 3])  # not tag_Xs
        multi_boxplot_rect(df, tag_grp, tag_ext,
                           figname=f'{figname}_dim2', annotX=lbl_dim2)
        multi_boxplot_rect(df, tag_grp, tag_ext, tag_ext_alt,
                           figname=f'{figname}_dim3', annotX=lbl_dim2)
        pdb.set_trace()
        return

    def avg_draw_extended_grp_scat(self, df, tag_grp, tag_ext, tag_ext_alt,
                                   figname, verbose=False, ver_mark=' (avg.)'):
        labels = ['ori', 'ext', 'alt', f'ext{ver_mark}', f'alt{ver_mark}']
        lbl_dim2 = GRP_FAIR_COMMON + [
            r'$\mathbf{df}_\text{prev}$', r'$\hat{\mathbf{df}}_\text{prev}$']
        # palette = ['black', '#387EB8', '#C24E44', '#184879', '#762E29']
        # palette_alt = ['black', '#066190', '#C42238', '#024163', '#8E0F31']
        # palette_whole = ['black', '#066190', '#C42238', '#024163', '#8E0F31',
        #                  '#387EB8', '#C24E44', '#184879', '#762E29', ]
        palette_whole = ['black', '#066190', '#C42238', '#024163', '#8E0F31',
                         '#77AECD', '#D98380', '#066190', '#C42238', ]
        palette = palette_whole[:5]
        palette_alt = ['black'] + palette_whole[-4:]
        labels = GRP_EXTENSIONS

        fgn = figname.replace('_avg', '')
        multi_boxplot_rect_revised(
            df, tag_grp[:3], tag_ext[:3], labels =['ori'] + labels[-2:],
            palette=palette_alt,      # figname=f'{fgn}_grpext_avg',
            figname=f'{fgn}_ge_avg', annotX=lbl_dim2[:3], locate="upper left")
        multi_boxplot_rect_revised(
            df, tag_grp[:3], tag_ext[:3], tag_ext_alt[:3],
            labels =['ori'] + labels[-2:], palette=palette_alt,
            figname=f'{fgn}_ga_avg',  # figname=f'{fgn}_grpalt_avg',
            annotX=lbl_dim2[:3], locate="upper left")
        multi_boxplot_rect_revised(
            df, tag_grp[:3], tag_ext[-3:], tag_ext_alt[-3:],
            tag_ext[:3], tag_ext_alt[:3], palette=palette_whole,
            labels=labels,
            figname=f'{fgn}_gm_avg',  # figname=f'{fgn}_group_max_avg',
            annotX=lbl_dim2[:3], locate="upper left", figsize='M-NT')

        multi_boxplot_rect_revised(
            df, tag_grp[:3], tag_ext[-3:], labels =labels[:3],
            figname=f'{fgn}_ge',  # figname=f'{fgn}_grpext',
            palette=palette, annotX=lbl_dim2[:3], locate="upper left")
        multi_boxplot_rect_revised(
            df, tag_grp[:3], tag_ext[-3:], tag_ext_alt[-3:],
            labels =labels[:3], figname=f'{fgn}_ga',
            # figname=f'{fgn}_grpalt',  # f'{fgn}_grpalt_max',
            palette=palette, annotX=lbl_dim2[:3], locate="upper left")
        os.remove(f'{fgn}_ge_avg.pdf')
        os.remove(f'{fgn}_ge.pdf')
        return

    def avg_depict_separately(self, pick_set, pick_clf, df, id_set,
                              tag_mk='tst', fgn='', multival=True):
        tag_acc, tag_sa1, tag_sa2 = self.obtain_tag_col(tag_mk)
        tag_acc = tag_acc[: -2]
        sub_grp = tag_sa1[:3] + [tag_sa1[16 + 3], tag_sa1[27 + 3]]
        sub_ext = tag_sa1[4:10][:3] + [tag_sa1[16 + 7], tag_sa1[27 + 7]]
        sub_ext_alt = tag_sa1[10:16][:3] + [tag_sa1[
            16 + 10], tag_sa1[27 + 10]]
        sub_idv = tag_acc[16:16 + 4 + 6]  # dr 4+ GEI.alph 3+ Theil+Tx2
        sub_idv = [sub_idv[2], ] + sub_idv[4:-2]
        sub_idv = [sub_idv[0], sub_idv[2], sub_idv[-1]]

        sub_ext_avg = tag_sa1[4:10][-3:] + [tag_sa1[23], tag_sa1[34]]
        sub_ext_alt_avg = tag_sa1[10:16][-3:] + [
            tag_sa1[26], tag_sa1[37]]
        currX = sub_grp[:3] + sub_idv + sub_grp[-2:]
        labels = GRP_FAIR_COMMON + [
            'DR', 'GEI', 'Theil',  # 'DR', r'GEI ($\alpha$=0.5)', 'Theil',
            r'$\mathbf{df}_{prev}$', r'$\hat{\mathbf{df}}_{prev}$']
        annotY = GRP_EXTENSIONS

        def _internal(ps, pc):
            df_alt = self.obtain_sing_dat_cls(
                ps, pc, tag_acc, tag_sa1, tag_sa2, df, id_set, multival)
            df_tmp = df_alt[currX]
            for i in currX:
                df_tmp.loc[:, i] = float(df_tmp[i].mean())
            df_tmp = df_tmp.reset_index(drop=True)
            df_tmp_tmp = df_alt[sub_ext]
            for i, j in zip(sub_grp, sub_ext):
                df_tmp.loc[1, i] = float(df_tmp_tmp[j].mean())
            df_tmp_tmp = df_alt[sub_ext_alt]
            for i, j in zip(sub_grp, sub_ext_alt):
                df_tmp.loc[2, i] = float(df_tmp_tmp[j].mean())
            df_tmp_tmp = df_alt[sub_ext_avg]
            for i, j in zip(sub_grp, sub_ext_avg):
                df_tmp.loc[3, i] = float(df_tmp_tmp[j].mean())
            df_tmp_tmp = df_alt[sub_ext_alt_avg]
            for i, j in zip(sub_grp, sub_ext_alt_avg):
                df_tmp.loc[4, i] = float(df_tmp_tmp[j].mean())
            return df_tmp, df_alt[currX]

        nm_set = ['ricci', 'credit', 'income', 'ppr', 'ppvr']
        nm_clf = ['bagging', 'AdaBoost', 'LightGBM',
                  'FairGBM', 'FairGBM', 'FairGBM', 'AdaFair#1',
                  'FairGBM', 'FairGBM', 'FairGBM', 'AdaFair#2']
        # '' '
        # for ps in pick_set:
        #     for pc in pick_clf:
        #         df_tmp, _ = _internal(ps, pc)
        #         radar_chart(df_tmp, currX, labels,  # annotY,
        #                     annotY if pc == 10 else None,
        #                     figname=f'{fgn}_s{ps}c{pc}', stylish=True)
        #         if pc > 3:
        #             continue
        #         tabular_chart(
        #             df_tmp, currX, labels, annotY,
        #             data=nm_set[ps], algo=nm_clf[pc],
        #             figname=f'{fgn.replace("radar", "tab")}_s{ps}c{pc}p')
        #         # os.remove(f'{fgn}_s{ps}c{pc}.pdf')
        #         # os.remove(f'{fgn.replace("radar", "tab")}_s{ps}c{pc}p.pdf')
        # '' '

        # '' '
        df_tmp_set = []
        for ps in pick_set:
            tmp = []
            for pc in pick_clf:
                tmp.append(_internal(ps, pc)[0])
            df_tmp_set.append(tmp)
            del tmp
        # # labels[4] = r'GEI ($\alpha$=0.5)'
        # currX = currX[-1:] + currX[:-1]
        # labels = labels[-1:] + labels[:-1]
        tabular_chart_gather([i[:3] for i in df_tmp_set], currX, labels, annotY,
                             data=[nm_set[ps] for ps in pick_set],
                             algo=[nm_clf[pc] for pc in pick_clf[:3]],
                             figname=f'{fgn.replace("radar","tab")}_scp',
                             panel='yp')
        radar_chart_gather(df_tmp_set, currX, labels, annotY,
                           figname=f'{fgn}_sc', stylish=True, sharey=True,
                           entitle=False)
        # '' '
        return

    def avg_draw_trade_off_alt(self, df, pick, tag_X, tag_Ys, figname):
        annotZs = GRP_FAIR_COMMON + [
            r'GEI ($\alpha$=0.5)', 'Theil', 'DR']
        annotZs[-3] = 'GEI'
        key_A = [tag_X[:8][i] for i in pick]
        key_C = [tag_X[8:16][i] for i in pick]
        key_B_bin = tag_Ys[0][:3] + tag_X[-3:] + tag_Ys[0][-2:]
        # lbl_A = [self._perf_metric[i] for i in pick]
        lbl_C = [self._dal_metric[i] for i in pick]
        lbl_B_bin = annotZs + [r'$\mathbf{df}_\text{prev}$',
                               r'$\hat{\mathbf{df}}_\text{prev}$']
        Mat_B_bin = df[key_B_bin].values.astype(DTY_FLT).T
        kws = {'cmap_name': 'Blues', 'rotate': 65}

        key_B_nonbin = tag_Ys[1][:3] + tag_Ys[3][:3] + tag_Ys[1][-2:]
        key_B_extalt = tag_Ys[2][:3] + tag_Ys[4][:3] + tag_Ys[2][-2:]
        Mat_B_ext = df[key_B_nonbin].values.astype(DTY_FLT).T
        Mat_B_extalt = df[key_B_extalt].values.astype(DTY_FLT).T

        # lbl_B_ext = [r'$\text{DP}^\text{ext}$',
        #              r'$\text{EOpp}^\text{ext}$',
        #              r'$\text{PP}^\text{ext}$',
        #              r'$\text{DP}^\text{ext(avg)}$',
        #              r'$\text{EOpp}^\text{ext(avg)}$',
        #              r'$\text{PP}^\text{ext(avg)}$', ] + [
        #     r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        # lbl_B_extalt = [r'$\text{DP}^\text{alt}$',
        #                 r'$\text{EOpp}^\text{alt}$',
        #                 r'$\text{PP}^\text{alt}$',
        #                 r'$\text{DP}^\text{alt(avg)}$',
        #                 r'$\text{EOpp}^\text{alt(avg)}$',
        #                 r'$\text{PP}^\text{alt(avg)}$', ] + [
        #     r'$\mathbf{df}^\text{avg}$',
        #     r'$\hat{\mathbf{df}}^\text{avg}$']
        lbl_B_ext = GRP_FAIR_COMMON + [
            '$\\Delta$DP$^\\text{avg}$',
            '$\\Delta$EOpp$^\\text{avg}$',
            '$\\Delta$PP$^\\text{avg}$'] + [
            r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        lbl_B_extalt = GRP_FAIR_COMMON + [
            '$\\Delta\\text{DP}^\\text{avg}$',
            '$\\Delta\\text{EOpp}^\\text{avg}$',
            '$\\Delta\\text{PP}^\\text{avg}$'] + [
            r'$\mathbf{df}^\text{avg}$',
            r'$\hat{\mathbf{df}}^\text{avg}$']

        fgn = figname[:-4]
        # os.remove(f'{fgn}_cont1p.pdf')
        # os.remove(f'{fgn}_cont2p.pdf')
        # os.remove(f'{fgn}_cont3p.pdf')
        # os.remove(f'{fgn}_avg_cont2p.pdf')
        # os.remove(f'{fgn}_avg_cont3p.pdf')
        Mat_C_A = df[key_C + key_A].values.astype(DTY_FLT).T

        tmp_C = unique_column(12 + 158 * 2)[-158:]
        tmp_C = [tmp_C[8:16][i] for i in [2, 3]]
        tmp_A = [tag_X[:8][i] for i in [2, 3]]
        Mat_C_A[7] = ((df[tmp_A[0]] + df[tmp_A[1]]) / 2).values.astype(DTY_FLT)
        Mat_C_A[3] = ((df[tmp_C[0]] + df[tmp_C[1]]) / 2).values.astype(DTY_FLT)
        Mat_C_A[3] = np.abs(Mat_C_A[7] - Mat_C_A[3])
        # pdb.set_trace()
        kws['rotate'] = 45  # Mat_C_A = Mat_C_A[: len(pick)]
        # 'blue','red,green'  # '#3162A1', '#D23B3E', '#F36F3E'  # '#9a0019'
        kws['cmap_name'] = 'gray'

        analogous_confusion_extended(
            Mat_C_A[:len(pick)], Mat_B_bin, lbl_C, lbl_B_bin,
            # Mat_C_A, Mat_B_bin, lbl_C + lbl_A, lbl_B_bin,
            f'{fgn}_cont1p', **kws)

        kws['entitle'] = 'Extended & Extended (avg.) group fairness, maximal HFM'
        kws['cmap_name'] = '#066190'  # 'Oranges'
        analogous_confusion_extended(
            Mat_C_A[:len(pick)], Mat_B_ext, lbl_C, lbl_B_ext,
            # Mat_C_A, Mat_B_ext, lbl_C + lbl_A, lbl_B_ext,
            f'{figname}_cont2p', **kws)
        kws['entitle'] = 'Alternative & Alternative (avg.) group fairness, average HFM'
        kws['cmap_name'] = '#C42238'  # 'RdPu'
        analogous_confusion_extended(
            Mat_C_A[:len(pick)], Mat_B_extalt, lbl_C,
            # Mat_C_A, Mat_B_extalt, lbl_C + lbl_A,
            lbl_B_extalt, f'{figname}_cont3p', **kws)
        # pdb.set_trace()
        os.remove(f'{fgn}_cont1p' + '.pdf')
        os.remove(f'{figname}_cont2p' + '.pdf')
        os.remove(f'{figname}_cont3p' + '.pdf')

        Mat_C_A = Mat_C_A[: len(pick)]
        key_D = key_B_bin[:3] + key_B_nonbin[:6] + key_B_extalt[:6]
        key_D = np.array(key_D).reshape(-1, 3).T.reshape(-1).tolist()
        lbl_D = ['Original', 'Extended', 'Extended\n (avg.)', 'Alternative',
                 'Alternative\n (avg.)']  # GRP_EXTENSIONS
        figname = figname.replace('_to_alt', '_tolc')  # _cont
        lbl_D1 = ['(A3)', '(A5)', '(A7)', '(A6)', '(A8)']
        lbl_D2 = ['(A9)', '(A5\')', '(A7\')', '(A6\')', '(A8\')']
        lbl_D3 = ['(A12)', '(A5\')', '(A7\')', '(A6\')', '(A8\')']
        anal_conf_extended_subplt(Mat_C_A, [
            df[key_D[:5]].values.astype(DTY_FLT).T,
            df[key_D[5:10]].values.astype(DTY_FLT).T,
            df[key_D[10:]].values.astype(DTY_FLT).T],
            lbl_C, [lbl_D1, lbl_D2, lbl_D3], f'{figname}_sep0',
            # key_tit=[i[8:] + ' & its extensions' for i in GRP_FAIR_COMMON],
            key_tit=[i + ' & its extensions' for i in GRP_FAIR_COMMON],
            cmap_name='#FED477')  # 'darkgray')  # '#F4870B')

        kws = {'rotate': 0, 'entitle': None, 'figsize': 'M-WS',
               'cmap_name': '#F4870B'}  # 'darkgray','#e64b35','#f39b7f'
        # # kws['entitle'] = '  DP & its extended formulations'
        # analogous_confusion_extended(
        #     Mat_C_A, df[key_D[:5]].values.astype(DTY_FLT).T,
        #     lbl_C, lbl_D1, f'{figname}_sep1', **kws)
        # # kws['entitle'] = 'EOpp & its extended formulations'
        # analogous_confusion_extended(
        #     Mat_C_A, df[key_D[5:10]].values.astype(DTY_FLT).T,
        #     lbl_C, lbl_D2, f'{figname}_sep2', **kws)
        # # kws['entitle'] = '  PP & its extended formulations'
        # analogous_confusion_extended(
        #     Mat_C_A, df[key_D[10:]].values.astype(DTY_FLT).T,
        #     lbl_C, lbl_D3, f'{figname}_sep3', **kws)
        key_D = key_B_bin[-2:] + key_B_nonbin[-2:] + key_B_extalt[-2:]
        # key_D = np.array(key_D).reshape(-1, 2).T.reshape(-1).tolist()
        lbl_D = lbl_B_bin[-2:] + lbl_B_ext[-2:] + lbl_B_extalt[-2:]
        # lbl_D = np.array(lbl_D).reshape(-1, 2).T.reshape(-1).tolist()
        # analogous_confusion_extended(
        #     Mat_C_A, df[key_D].values.astype(DTY_FLT).T,
        #     lbl_C, lbl_D, f'{figname}_sep5', **kws)
        # lbl_D[0] = '\n' + lbl_D[0]
        # lbl_D[2] = '\n' + lbl_D[2]
        # lbl_D[4] = '\n' + lbl_D[4]
        lbl_D[1] = '    ' + lbl_D[1]
        lbl_D[0] = '  ' + lbl_D[0]
        key_E = key_B_bin[3:6] + key_D[2:] + key_D[:2]
        lbl_E = lbl_B_bin[3:6] + lbl_D[2:] + lbl_D[:2]
        kws['figsize'] = 'L-WT' if len(key_E) > 7 else 'L-NT'
        # kws['rotate'] = 55
        kws['cmap_name'] = '#00a087'  # 'gray', '#3162A1'
        # analogous_confusion_extended(
        #     Mat_C_A, df[key_E].values.astype(DTY_FLT).T,
        #     lbl_C, lbl_E, f'{figname}_sep4', **kws)
        # # key_E = key_B_bin[:3] + key_D[:2]
        # # lbl_E = lbl_B_bin[:3] + lbl_D[:2]
        # # analogous_confusion_extended(
        # #     Mat_C_A, df[key_E].values.astype(DTY_FLT).T,
        # #     lbl_C, lbl_E, f'{figname}_sep5', **kws)

        anal_conf_extended_subplt(Mat_C_A, [
            df[key_E[:3]].values.astype(DTY_FLT).T,
            df[key_E[-2:]].values.astype(DTY_FLT).T,
            df[key_E[3:-2]].values.astype(DTY_FLT).T],
            lbl_C, [lbl_E[:3], lbl_E[-2:], lbl_E[3:-2]],
            f'{figname}_sep6', cmap_name=kws['cmap_name'],
            figsize=(7.8, 1.91), key_tit='Individual fairness & HFM')
        return

    def avg_draw_incompatible_alt(self, df, tag_X, tag_Ys, figname):
        #                         , verbose=False):
        annotZs = GRP_FAIR_COMMON + [r'GEI ($\alpha$=0.5)', 'Theil', 'DR']
        ra = f"{'':>8}"  # ta, ra = f"{'':<7}", f"{'':>8}"
        # tmp_ext = [r'$\text{DP}^\text{ext}$' + ta,
        #            r'$\text{EOpp}^\text{ext}$' + ta,
        #            r'$\text{PP}^\text{ext}$' + ta, ]
        # tmp_ext_alt = [r'$\text{DP}^\text{alt}$' + ra,
        #                r'$\text{EOpp}^\text{alt}$' + ra,
        #                r'$\text{PP}^\text{alt}$' + ra, ]
        # tmp_ext_avg = [r'$\text{DP}^\text{ext(avg)}$',
        #                r'$\text{EOpp}^\text{ext(avg)}$',
        #                r'$\text{PP}^\text{ext(avg)}$']
        # tmp_ext_alt_avg = [r'$\text{DP}^\text{alt(avg)}$' + ' ',
        #                    r'$\text{EOpp}^\text{alt(avg)}$' + ' ',
        #                    r'$\text{PP}^\text{alt(avg)}$' + ' ']
        annotZs[0] += ra + f"{'':>3}"
        annotZs[2] += ra + f"{'':>2}"
        annotZs[1] += ra + f"{'':<3}"
        grp1 = [y[0] for y in tag_Ys]
        grp2 = [y[1] for y in tag_Ys]
        grp3 = [y[2] for y in tag_Ys]
        # lbl_g1 = [annotZs[0], tmp_ext[0], tmp_ext_alt[0],
        #           tmp_ext_avg[0], tmp_ext_alt_avg[0]]
        # lbl_g2 = [annotZs[1], tmp_ext[1], tmp_ext_alt[1],
        #           tmp_ext_avg[1], tmp_ext_alt_avg[1]]
        # lbl_g3 = [annotZs[2], tmp_ext[2], tmp_ext_alt[2],
        #           tmp_ext_avg[2], tmp_ext_alt_avg[2]]
        lbl_g = [f'Original{"":15s}',
                 f'Extended{"":12s}', f'Alternative{"":10s}',
                 f'Extended (avg.){"":2s}', 'Alternative (avg.)']
        kws = {'snspec': 'sty5b'}  # {},'invt_a':True

        # '' '
        # for i, pk in enumerate(tag_X[-3:]):
        #     annotX, fgn = annotZs[i + 3], f'{figname}_corr{i+4}'
        #     linreg_w_marg_dist_revised(
        #         df, pk, 'Fairness', grp1, lbl_g, annotX=annotX,
        #         annotY=GRP_FAIR_COMMON[0], figname=f'{fgn}_grp1', **kws)
        #     linreg_w_marg_dist_revised(
        #         df, pk, 'Fairness', grp2, lbl_g, annotX=annotX,
        #         annotY=GRP_FAIR_COMMON[1], figname=f'{fgn}_grp2', **kws)
        #     linreg_w_marg_dist_revised(
        #         df, pk, 'Fairness', grp3, lbl_g, annotX=annotX,
        #         annotY=GRP_FAIR_COMMON[2], figname=f'{fgn}_grp3',
        #         curr_key=True, **kws)
        # pdb.set_trace()
        # ' ''
        annotZs[-3] = 'GEI'
        linreg_w_marg_dist_rev_sup_pv2(
            df, 'Fairness', [-2, -1, -3], tag_X[-3:], [grp1, grp2, grp3],
            lbl_g, antX=annotZs[-3:], antYs=GRP_FAIR_COMMON,
            figname=f'{figname}_corr_grp', **kws)  # gap=True,

        hfm_drt = [tag_Ys[0][-2], tag_Ys[1][-2], tag_Ys[2][-2]]
        hfm_app = [tag_Ys[0][-1], tag_Ys[1][-1], tag_Ys[2][-1]]
        lbl_drt = [r'$\mathbf{df}_\text{prev}$', r'$\mathbf{df}$',
                   r'$\mathbf{df}^\text{avg}$']
        lbl_app = [r'$\hat{\mathbf{df}}_\text{prev}$',
                   r'$\hat{\mathbf{df}}$',
                   r'$\hat{\mathbf{df}}^\text{avg}$']
        # ' ''
        # for k, (pi, pj) in enumerate(zip(hfm_drt, hfm_app)):
        #     fgn = f'{figname}_df{k}'
        #     linreg_w_marg_dist_revised(
        #         df, pi, 'Fairness', grp1, lbl_g, annotX=lbl_drt[k],
        #         annotY=GRP_FAIR_COMMON[0], figname=f'{fgn}_g1d', **kws)
        #     linreg_w_marg_dist_revised(
        #         df, pi, 'Fairness', grp2, lbl_g, annotX=lbl_drt[k],
        #         annotY=GRP_FAIR_COMMON[1], figname=f'{fgn}_g2d', **kws)
        #     linreg_w_marg_dist_revised(
        #         df, pi, 'Fairness', grp3, lbl_g, annotX=lbl_drt[k],
        #         annotY=GRP_FAIR_COMMON[2], figname=f'{fgn}_g3d',
        #         curr_key=True, **kws)
        #     # if not verbose:
        #     #     continue
        #     # linreg_w_marg_dist_revised(
        #     #     df, pj, 'Fairness', grp1, lbl_g1, annotX=lbl_app[k],
        #     #     annotY=lbl_g1[0], figname=f'{fgn}_g1a', **kws)
        #     # linreg_w_marg_dist_revised(
        #     #     df, pj, 'Fairness', grp2, lbl_g2, annotX=lbl_app[k],
        #     #     annotY=lbl_g2[0], figname=f'{fgn}_g2a', **kws)
        #     # linreg_w_marg_dist_revised(
        #     #     df, pj, 'Fairness', grp3, lbl_g3, annotX=lbl_app[k],
        #     #     annotY=lbl_g3[0], figname=f'{fgn}_g3a', **kws)
        # '' '
        linreg_w_marg_dist_rev_sup_pv2(
            df, 'Fairness', [0, 1, 2], hfm_drt, [grp1, grp2, grp3],
            lbl_g, antX=lbl_drt, antYs=GRP_FAIR_COMMON,
            figname=f'{figname}_df_gd', **kws)
        linreg_w_marg_dist_rev_sup_pv2(
            df, 'Fairness', [0, 1, 2], hfm_app, [grp1, grp2, grp3],
            lbl_g, antX=lbl_app, antYs=GRP_FAIR_COMMON,
            figname=f'{figname}_df_ga', start_pt_ind=9, **kws)
        return

    def avg_draw_trade_off(self, df, pick, tag_X, tag_Ys, figname, ver_mark=''):
        # labels = ['Original', 'Extended', 'Alternative',
        #           f'Extended{ver_mark}', f'Alternative{ver_mark}']
        # # lbl_dim2 = GRP_FAIR_COMMON + [
        # #     r'$\mathbf{df}_\text{prev}$', r'$\hat{\mathbf{df}}_\text{prev}$']
        # # fgn = figname.replace('_avg', '')

        # # annotZs = GRP_FAIR_COMMON + [r'GEI ($\alpha$=0.5)', 'Theil', 'DR']
        annotY = 'Extended (avg.) GF (multival)'  # 'group fairness (multival)'
        # # tmp_ext = [r'$\text{DP}^\text{ext(avg)}$',
        # #            r'$\text{EOpp}^\text{ext(avg)}$',
        # #            r'$\text{PP}^\text{ext(avg)}$', ]
        # # tmp_ext_alt = [r'$\text{DP}^\text{alt(avg)}$',
        # #                r'$\text{EOpp}^\text{alt(avg)}$',
        # #                r'$\text{PP}^\text{alt(avg)}$', ]
        # tmp_ext = tmp_ext_alt = GRP_FAIR_COMMON

        tw = {'snspec': 'sty4b', 'palette_X': ['#FED477'] * 3,
              'palette_Y': ['#00a087'] * 9}  # X:'#F4870B'  cde5cd,E6DAc3
        # # tw['palette_Y'] = ['#168E6A', ] + ['#1E827B', '#586395'] * 4
        # # tw['palette_Y'] = ['#ABC6DA'] + ['#6C9BD8', '#45924c'] * 4
        # # tw['palette_Y'] = ['#3f8680'] + ['#45924c', '#cae4e2'] * 4
        # # tw['palette_Y'] = ['#288f82'] + ['#55c47e', '#7cc94a'] * 4
        # tw['palette_Y'] = ['#285B90', ] + ['#377BAB', '#956A88'] * 4
        # tw['palette_Y'] = ['#483D8B', ] + ['#7367BE', '#B1A9DA'] * 4
        # tw['palette_Y'] = ['#8EA0CC', ] + ['#8EA0CC', '#CCB4D7'] * 4
        # tw['palette_Y'] = ['#377BAB', ] + ['#377BAB', '#956A88'] * 4
        # tw['palette_Y'] = ['#483D8B', ] + ['#483D8B', '#7367BE'] * 4
        tw['palette_Y'] = ['#501d8a', ] + ['#aa3474', '#ee8c7d'] * 4
        # for pk in pick:
        #     annotX = self._perf_metric[pk]
        #     if pk == 6:
        #         tw['curr_key'] = True
        #     linreg_w_marg_dist_revised(
        #         df, tag_X[pk], 'Fairness', tag_Ys[1][:3],
        #         tmp_ext, annotX=annotX, annotY=annotY,
        #         figname=f'{figname}_to{pk}_s2', **tw)
        #     linreg_w_marg_dist_revised(
        #         df, tag_X[pk], 'Fairness', tag_Ys[2][:3],
        #         tmp_ext_alt, annotX=annotX, annotY=annotY.replace(
        #             'Extended', 'Alternative extended'),
        #         figname=f'{figname}_to{pk}_s3', **tw)
        # # line_reg_with_marginal_distr(snspec='sty4b',
        linreg_w_marg_dist_rev_sup_pv1(
            df, 'Fairness', pick, tag_X, [tag_Ys[i][:3] for i in [1, 2]],
            GRP_FAIR_COMMON, antX=self._perf_metric,  # ]*len(pick),
            antYs=[annotY, annotY.replace('Extended', 'Alternative')],
            figname=f'{figname[:-4]}_ps_avg',  # f'{figname}_to_ps'
            start_pt_ind=8, subfig=True, **tw)

        # ' ''
        # key_A = [tag_X[:8][i] for i in pick]
        # key_C = [tag_X[8:16][i] for i in pick]
        # key_B_nonbin = tag_Ys[1][:3] + tag_X[-3:] + tag_Ys[1][-2:]
        # key_B_extalt = tag_Ys[2][:3] + tag_X[-3:] + tag_Ys[2][-2:]
        # lbl_A = [self._perf_metric[i] for i in pick]
        # lbl_C = [self._dal_metric[i] for i in pick]
        # lbl_B_bin = annotZs + [r'$\mathbf{df}_\text{prev}$',
        #                        r'$\hat{\mathbf{df}}_\text{prev}$']
        # # Mat_B_bin = df[key_B_bin].values.astype(DTY_FLT).T
        # Mat_B_ext = df[key_B_nonbin].values.astype(DTY_FLT).T
        # Mat_B_extalt = df[key_B_extalt].values.astype(DTY_FLT).T
        # kws = {'cmap_name': 'Blues', 'rotate': 65}
        #
        # Mat_C_A = df[key_C + key_A].values.astype(DTY_FLT).T
        # tmpC = [tag_X[:8][2], tag_X[:8][3]]
        # Mat_C_A[7] = ((df[tmpC[0]] + df[tmpC[1]]) / 2).values.astype(DTY_FLT)
        # tmpC = unique_column(12 + 158 * 2)[-158:][:24][8:-8][2:4]
        # Mat_C_A[3] = ((df[tmpC[0]] + df[tmpC[1]]) / 2).values.astype(DTY_FLT)
        # Mat_C_A[3] = np.abs(Mat_C_A[7] - Mat_C_A[3])
        # Mat_C_A = Mat_C_A[: len(key_C)]
        #
        # lbl_B_ext = tmp_ext + lbl_B_bin[3:6] + [
        #     r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        # lbl_B_extalt = tmp_ext_alt + lbl_B_bin[3:6] + [
        #     r'$\mathbf{df}^\text{avg}$',
        #     r'$\hat{\mathbf{df}}^\text{avg}$']
        # kws['cmap_name'] = '#066190'  # 'Oranges'
        # analogous_confusion_extended(
        #     Mat_C_A, Mat_B_ext,  # lbl_C + lbl_A,
        #     lbl_C, lbl_B_ext, f'{figname}_cont2p', **kws)
        # kws['cmap_name'] = '#C42238'  # 'RdPu'
        # analogous_confusion_extended(
        #     Mat_C_A, Mat_B_extalt,  # lbl_C + lbl_A,
        #     lbl_C, lbl_B_extalt, f'{figname}_cont3p', **kws)
        # ' ''
        return

    def draw_trade_off(self, df, pick, tag_X, tag_Ys, figname, ver_mark=''):
        annotZs = GRP_FAIR_COMMON + [
            r'GEI ($\alpha$=0.5)', 'Theil', 'DR']
        # tmp_ext = [r'$\text{DP}^\text{ext}$',
        #            r'$\text{EOpp}^\text{ext}$',
        #            r'$\text{PP}^\text{ext}$', ]
        # tmp_ext_alt = [r'$\text{DP}^\text{alt}$',
        #                r'$\text{EOpp}^\text{alt}$',
        #                r'$\text{PP}^\text{alt}$', ]

        # for pk in pick:
        #     annotX = self._perf_metric[pk]
        #     linreg_w_marg_dist_revised(  # tag_X[-2:]+tag_X[-3:-2]
        #         df, tag_X[pk], 'Fairness', tag_X[-3:], annotZs[-3:],
        #         annotX=annotX, annotY='Individual fairness',
        #         snspec='sty4b', figname=f'{figname}_to{pk}_s4')
        #     linreg_w_marg_dist_revised(
        #         df, tag_X[pk], 'Fairness', tag_Ys[0][:3], annotZs[:3],
        #         annotX=annotX, annotY='Group fairness (bin-val)',
        #         snspec='sty4b', figname=f'{figname}_to{pk}_s1')
        #     linreg_w_marg_dist_revised(
        #         df, tag_X[pk], 'Fairness', tag_Ys[1][:3],
        #         # [f'{i} ext.' for i in annotZs[:3]], annotX=annotX,
        #         tmp_ext, annotX=annotX,
        #         annotY='Extended group fairness (multival)',
        #         snspec='sty4b', figname=f'{figname}_to{pk}_s2')
        #     linreg_w_marg_dist_revised(
        #         # df, tag_X[pk], 'Fairness', tag_Ys[0][:3],
        #         # [f'{i} ext. alt' for i in annotZs[:3]], annotX=annotX,
        #         df, tag_X[pk], 'Fairness', tag_Ys[2][:3],
        #         tmp_ext_alt, annotX=annotX,
        #         annotY='Alternative extended group fairness (multival)',
        #         snspec='sty4b', figname=f'{figname}_to{pk}_s3')
        # pdb.set_trace()
        annotY = 'Extended GP (multival)'  # line_reg_with_marginal_distr(
        tw = {'snspec': 'sty4b', 'subfig': True, 'palette_X': ['#FED477'] * 3,
              'palette_Y': ['#501d8a'] + ['#aa3474', '#ee8c7d'] * 4}
        annotZs[3] = 'GEI'
        # linreg_w_marg_dist_revised_subplt(
        #     df, 'Fairness', pick, tag_X,
        #     [tag_X[-3:], ] + [tag_Ys[i][:3] for i in [0, 1, 2]],
        #     GRP_FAIR_COMMON, antX=self._perf_metric, antYs=[
        #         'IF', 'GF (bin-val)', annotY, annotY.replace(
        #             'Extended', 'Alternative')], figname=f'{figname}_to_s', **tw)
        linreg_w_marg_dist_rev_sup_pv1(
            df, 'Fairness', pick, tag_X, [tag_Ys[0][:3], tag_X[-3:]],
            GRP_FAIR_COMMON + ['GEI'] + annotZs[-2:], antX=self._perf_metric,
            antYs=['GF (bin-val)', 'IF'], figname=f'{figname}_si', **tw)
        linreg_w_marg_dist_rev_sup_pv1(
            df, 'Fairness', pick, tag_X, [tag_Ys[i][:3] for i in [1, 2]],
            GRP_FAIR_COMMON, antX=self._perf_metric, antYs=[
                annotY, annotY.replace('Extended', 'Alternative')],
            figname=f'{figname}_ps', **tw)  # f'{figname}_max_se'  # _to_si

        # '' '
        # key_A = [tag_X[:8][i] for i in pick]
        # key_C = [tag_X[8:16][i] for i in pick]
        # key_B_bin = tag_Ys[0][:3] + tag_X[-3:] + tag_Ys[0][-2:]
        # key_B_nonbin = tag_Ys[1][:3] + tag_X[-3:] + tag_Ys[1][-2:]
        # key_B_extalt = tag_Ys[2][:3] + tag_X[-3:] + tag_Ys[2][-2:]
        # lbl_A = [self._perf_metric[i] for i in pick]
        # lbl_C = [self._dal_metric[i] for i in pick]
        # lbl_B_bin = annotZs + [r'$\mathbf{df}_\text{prev}$',
        #                        r'$\hat{\mathbf{df}}_\text{prev}$']
        # lbl_B_ext = tmp_ext + lbl_B_bin[3:6] + [
        #     r'$\mathbf{df}$', r'$\hat{\mathbf{df}}$']
        # lbl_B_extalt = tmp_ext_alt + lbl_B_bin[3:6] + [
        #     r'$\mathbf{df}^\text{avg}$',
        #     r'$\hat{\mathbf{df}}^\text{avg}$']
        # Mat_B_bin = df[key_B_bin].values.astype(DTY_FLT).T
        # Mat_B_ext = df[key_B_nonbin].values.astype(DTY_FLT).T
        # Mat_B_extalt = df[key_B_extalt].values.astype(DTY_FLT).T
        # kws = {'cmap_name': 'dimgray', 'rotate': 65}  # 'Blues'
        # analogous_confusion_extended(
        #     # df[key_C + key_A].values.astype(DTY_FLT).T, Mat_B_bin,
        #     # lbl_C + lbl_A, lbl_B_bin, f'{figname}_cont1p', **kws)
        #     df[key_C].values.astype(DTY_FLT).T, Mat_B_bin,
        #     lbl_C, lbl_B_bin, f'{figname}_cont1p', **kws)
        # kws['cmap_name'] = '#066190'  # 'Oranges'
        # analogous_confusion_extended(
        #     # df[key_C + key_A].values.astype(DTY_FLT).T, Mat_B_ext,
        #     # lbl_C + lbl_A, lbl_B_ext, f'{figname}_cont2p', **kws)
        #     df[key_C].values.astype(DTY_FLT).T, Mat_B_ext,
        #     lbl_C, lbl_B_ext, f'{figname}_cont2p', **kws)
        # kws['cmap_name'] = '#C42238'  # 'RdPu'
        # analogous_confusion_extended(
        #     # df[key_C + key_A].values.astype(DTY_FLT).T, Mat_B_extalt,
        #     # lbl_C + lbl_A, lbl_B_extalt, f'{figname}_cont3p', **kws)
        #     df[key_C].values.astype(DTY_FLT).T, Mat_B_extalt,
        #     lbl_C, lbl_B_extalt, f'{figname}_cont3p', **kws)
        # '' '
        return

    def schedule_mspaint(self, raw_dframe, figname=''):
        _, id_set = self.recap_sub_data(raw_dframe, sa_ir=3, sa_r=4)
        mk = 'tst'  # nb_set,id_set=self.recap_sub_data(
        first_incl = verbose = False
        df_nonbin = self.obtain_multival_senatt(
            raw_dframe, id_set, mk, first_incl=first_incl)
        tag_acc, tag_sa1, _ = self.obtain_tag_col(mk)

        tmp = tag_sa1[-6: -3]
        df_nonbin['extGrp'] = df_nonbin[
            tmp[0]] + df_nonbin[tmp[1]] + df_nonbin[tmp[2]]
        tmp = tag_sa1[-3:]
        df_nonbin['extAlt'] = df_nonbin[tmp[0]] + df_nonbin[
            tmp[1]] + df_nonbin[tmp[2]] + df_nonbin['extGrp']
        pick = [0, 4, 5, 6]  # [0, 4, 5]

        curr = tag_acc[:8][2:4] + tag_acc[:8][6:7]
        df_nonbin[curr[2]] = (df_nonbin[curr[0]] + df_nonbin[curr[1]]) / 2
        curr = tag_acc[-2:] + tag_acc[8:16][6:7] + curr[-1:]  # tag_acc[8:16][2:4]
        df_nonbin[curr[2]] = (df_nonbin[curr[0]] + df_nonbin[curr[1]]) / 2
        df_nonbin[curr[2]] = np.abs(df_nonbin[curr[3]] - df_nonbin[curr[2]])
        del curr  # pdb.set_trace()
        tag_acc = tag_acc[: -2]

        col_grp = tag_sa1[:3] + [tag_sa1[16 + 3], tag_sa1[27 + 3]]
        col_ext = tag_sa1[4:10][:3] + [tag_sa1[16 + 7], tag_sa1[27 + 7]]
        col_ext_alt = tag_sa1[10:16][:3] + [
            tag_sa1[16 + 10], tag_sa1[27 + 10]]
        # self.draw_extended_grp_scat(
        #     df_nonbin, col_grp, col_ext, col_ext_alt,
        #     f'{figname}_scat', verbose)
        if verbose:
            self.draw_trade_off(df_nonbin, pick, tag_acc[:16] + [
                tag_acc[19 + 2], tag_acc[19 + 4], tag_acc[15 + 3], ], [
                col_grp, col_ext, col_ext_alt], f'{figname}_to')

        tim_idv = [tag_acc[16 + 3], ] + tag_acc[16 + 4 + 4:][:2]
        tim_idv = tim_idv[:: -1]   # DR,Theil,GEI: then reverse
        tim_df_pl = [tag_acc[26:][6], tag_acc[26:][6 + 7],
                     ]  # df/hat_df multiver (df intersectional)
        tim_grp = [tag_sa1[3], tag_sa1[16], 'extGrp', 'extAlt']  # three
        tim_df = [tag_sa1[20], tag_sa1[27], tag_sa1[27 + 4], tag_sa1[
            27 + 4 + 7]]  # df4one sen-att: bin-val, multival, hat_df x2
        df_nonbin['idv_ptb'] = df_nonbin[tag_acc[-2]] + df_nonbin[
            tim_idv[-1]]  # 'idvDR_perturb','idv_dr_ptb', 'idvDR_'
        # '' '
        # self.draw_extended_idv_tim(df_nonbin, tim_grp[:2], [
        #     tim_idv, tim_df, tim_df_pl], figname + '_idv')
        # self.draw_extended_grp_tim(df_nonbin, tag_sa1[3], [
        #     tag_sa1[16], 'extGrp', 'extAlt'], figname + '_grp')
        # self.draw_extended_hfm_tim(df_nonbin, tag_sa1[20], [
        #     tag_sa1[27], tag_sa1[27 + 4], tag_sa1[27 + 4 + 7]],
        #     figname + '_hfm')
        # # pdb.set_trace()
        # '' '

        tag_idv = (tim_grp[:2], [tim_idv, tim_df, tim_df_pl])
        tag_grp = (tag_sa1[3], [tag_sa1[16], 'extGrp', 'extAlt'])
        tag_hfm = (tag_sa1[20], [
            tag_sa1[27], tag_sa1[27 + 4], tag_sa1[27 + 4 + 7]])
        self.avg_draw_extended_tim(df_nonbin, tag_grp, tag_idv, tag_hfm,
                                   figname + '_collect_tim', verbose)
        del tag_grp, tag_idv, tag_hfm
        return

    def schedule_mspaint_avg(self, raw_dframe, figname=''):
        _, id_set = self.recap_sub_data(raw_dframe, sa_ir=3, sa_r=4)
        mk, first_incl, verbose = 'tst', False, False
        df_nonbin = self.obtain_multival_senatt(
            raw_dframe, id_set, mk, first_incl=first_incl)
        tag_acc, tag_sa1, _ = self.obtain_tag_col(mk)
        tmp = tag_sa1[-6: -3]
        df_nonbin['extGrp'] = df_nonbin[tmp[
            0]] + df_nonbin[tmp[1]] + df_nonbin[tmp[2]]    # TimeCost
        tmp = tag_sa1[-3:]
        df_nonbin['extAlt'] = df_nonbin[tmp[0]] + df_nonbin[tmp[
            1]] + df_nonbin[tmp[2]] + df_nonbin['extGrp']  # TimeCost
        pick = [0, 4, 5, 6]  # [0, 4, 5]

        curr = tag_acc[:8][2:4] + tag_acc[:8][6:7]
        df_nonbin[curr[2]] = (df_nonbin[curr[0]] + df_nonbin[curr[1]]) / 2
        curr = tag_acc[-2:] + tag_acc[8:16][6:7] + curr[-1:]  # tag_acc[8:16][2:4]
        df_nonbin[curr[2]] = (df_nonbin[curr[0]] + df_nonbin[curr[1]]) / 2
        df_nonbin[curr[2]] = np.abs(df_nonbin[curr[3]] - df_nonbin[curr[2]])
        del curr  # pdb.set_trace()
        tag_acc = tag_acc[: -2]

        # extension in average forms, above is maximal forms
        col_grp = tag_sa1[:3] + [tag_sa1[16 + 3], tag_sa1[27 + 3]]
        col_ext = tag_sa1[4:10][-3:] + [tag_sa1[23], tag_sa1[34]]
        col_ext_alt = tag_sa1[10:16][-3:] + [tag_sa1[26], tag_sa1[37]]
        if verbose:
            self.avg_draw_trade_off(df_nonbin, pick, tag_acc[
                :16] + [tag_acc[21], tag_acc[23], tag_acc[18], ], [
                col_grp, col_ext, col_ext_alt], f'{figname}_to_avg')
        self.avg_draw_extended_grp_scat(
            df_nonbin, col_grp, col_ext + tag_sa1[4:7],
            col_ext_alt + tag_sa1[10:13], f'{figname}_scat_avg', verbose)

        fgn = f'{figname}_radar_avg'
        # for pkc in [0, 1, 2, 6, 10]:
        #     for pks in [2, 3, 4]:
        #         self.avg_depict_separately(
        #             pks, pkc, raw_dframe, id_set, mk, fgn)
        #         if pkc == 2:
        #             continue
        #         os.remove(f'{fgn[:-4]}_s{pks}c{pkc}_ori.pdf')
        self.avg_depict_separately(
            [2, 3, 4], [0, 1, 2, 6, 10], raw_dframe, id_set, mk, fgn)
        #   # [2, 3, 4], [0, 1, 2], raw_dframe, id_set, mk, fgn)

        if not verbose:
            return
        col_ext_max = tag_sa1[4:10][:3] + [tag_sa1[23], tag_sa1[34]]
        col_ext_alt_max = tag_sa1[10:16][:3] + [
            tag_sa1[26], tag_sa1[37]]
        self.avg_draw_trade_off_alt(
            df_nonbin, pick, tag_acc[:16] + [
                tag_acc[21], tag_acc[23], tag_acc[18], ], [
                col_grp, col_ext_max, col_ext_alt_max,
                col_ext, col_ext_alt], f'{figname}_to_alt')
        self.avg_draw_incompatible_alt(
            df_nonbin, tag_acc[:16] + [
                tag_acc[21], tag_acc[23], tag_acc[18], ], [
                col_grp, col_ext_max, col_ext_alt_max,
                col_ext, col_ext_alt], f'{figname}_nc')
        return

    def avg_draw_extended_tim(self, df, tag_grp, tag_idv, tag_hfm,
                              figname, verbose=False):
        tg_X, tg_Ys = tag_grp
        ti_X, ti_Ys = tag_idv
        th_X, th_Ys = tag_hfm
        ant_X = [r'$T_\text{bin-val}$ (sec)', r'$T_\text{gf (bin-val)}$ (sec)',
                 r'$T_{\mathbf{df}_\text{prev} ~\text{(bin-val)}}$']
        ant_Ys = [r'$\frac{ T_\text{multival} }{ T_\text{bin-val} }$',
                  r'$\frac{ T_\text{if (multival)} }{ T_\text{gf (bin-val)} }$',
                  r'$\frac{ T_{\mathbf{df} ~\text{(multival)}} }{ T_{\mathbf{df}_\text{prev} ~\text{(bin-val)}} }$']
        ant_Zs = [r'$T_\text{multival} = T_\text{bin-val}$',
                  [r'$T_\text{if} = T_\text{gf (bin-val)}$',
                   r'$T_\text{gf (multival)}$', r'$T_\text{GEI}$', r'$T_\text{Theil}$',
                   r'$T_\text{DR}$'],
                  r'$T_{\mathbf{df} ~\text{(multival)}} = T_{\mathbf{df}_\text{prev} ~\text{(bin-val)}}$',
                  [r'$T_{\mathbf{df} ~\text{(multival)}} = T_{\mathbf{df}_\text{prev} ~\text{(bin-val)}}$',
                   r'$T_{\mathbf{df} ~\text{(multival)}}$',
                   r'$T_{\hat{\mathbf{df}} ~\text{(bin-val)}}$',
                   r'$T_{\hat{\mathbf{df}} ~\text{(multival)}}$']]
        # ant_X[0] = ant_X[1]  # r'$T_\text{gf (bin-val)}$ (sec)'
        # ant_Ys[0] = r'$\frac{ T_\text{gf (multival)} }{ T_\text{gf (bin-val)} }$'
        # ant_Zs[0] = r'$T_\text{gf (multival)} = T_\text{gf (bin-val)}$'
        ant_Zs[3][0] = r'$T_{\mathbf{df}} = T_{\mathbf{df}_\text{prev} ~\text{(bin-val)}}$'

        tmp = [ti_X[1], ] + ti_Ys[0]
        gathering_lin_reg_sup_tim(
            df, [tg_X, ti_X[0], th_X, th_X], [
                tg_Ys[0], tmp[:-1], th_Ys[0], th_Ys],
            figname, ant_X, ant_Ys, ant_Zs)
        # pdb.set_trace()
        return


class PlotB_gather(PlotB_fair_ens):
    # def data_gathering_alt(self, raw_df_b, raw_df_c, mk='tst'):
    #     first_incl = False
    #     nb_set, id_set = self.recap_sub_data(raw_df_b, sa_ir=3, sa_r=4)
    #     pdb.set_trace()
    #     nb_set, id_set = self.recap_sub_data(raw_df_c, sa_ir=11, sa_r=0)
    #     pdb.set_trace()
    #     return

    def data_gathering(self, raw_df_b, raw_df_c, mk='tst'):
        # first_incl = verbose = False     # mk = 'tst'
        first_incl = False
        _, id_set = self.recap_sub_data(raw_df_b, sa_ir=3, sa_r=4)
        df_b = self.obtain_multival_senatt(raw_df_b, id_set, mk, first_incl)
        _, id_set = self.recap_sub_data(raw_df_c, sa_ir=11, sa_r=0)
        df_c = self.obtain_multival_senatt(raw_df_c, id_set, mk, first_incl)
        df_nonbin = pd.concat([df_b, df_c], axis=0).reset_index(drop=True)
        # df_nonbin = df_b  # pdb.set_trace()

        tag_acc, tag_sa1, _ = self.obtain_tag_col(mk)
        curr = tag_acc[:8][2:4] + tag_acc[:8][6:7]
        df_nonbin[curr[2]] = (df_nonbin[curr[0]] + df_nonbin[curr[1]]) / 2
        curr = tag_acc[-2:] + tag_acc[8:16][6:7] + curr[-1:]
        df_nonbin[curr[2]] = (df_nonbin[curr[0]] + df_nonbin[curr[1]]) / 2
        df_nonbin[curr[2]] = np.abs(df_nonbin[curr[3]] - df_nonbin[curr[2]])
        del curr
        tag_acc = tag_acc[: -2]

        tmp = tag_sa1[-6: -3]
        df_nonbin['extGrp'] = df_nonbin[tmp[
            0]] + df_nonbin[tmp[1]] + df_nonbin[tmp[2]]    # TimeCost
        tmp = tag_sa1[-3:]
        df_nonbin['extAlt'] = df_nonbin[tmp[0]] + df_nonbin[tmp[
            1]] + df_nonbin[tmp[2]] + df_nonbin['extGrp']  # TimeCost
        return df_nonbin, tag_acc  # tt

    def schedule_mspaint(self, raw_dframe, figname=''):
        mk, verbose = 'tst', False
        tag_acc, tag_sa1, _ = self.obtain_tag_col(mk)
        df_nonbin, tag_acc = self.data_gathering(*raw_dframe)
        pick = [0, 4, 5, 6]

        col_grp = tag_sa1[:3] + [tag_sa1[16 + 3], tag_sa1[27 + 3]]
        col_ext = tag_sa1[4:10][:3] + [tag_sa1[16 + 7], tag_sa1[27 + 7]]
        col_ext_alt = tag_sa1[10:16][:3] + [
            tag_sa1[16 + 10], tag_sa1[27 + 10]]
        self.draw_trade_off(df_nonbin, pick, tag_acc[:16] + [
            tag_acc[19 + 2], tag_acc[19 + 4], tag_acc[15 + 3], ], [
            col_grp, col_ext, col_ext_alt], f'{figname}_to')
        tim_idv = [tag_acc[16 + 3], ] + tag_acc[16 + 4 + 4:][:2]
        tim_idv = tim_idv[:: -1]   # DR,Theil,GEI: then reverse
        tim_df_pl = [tag_acc[26:][6], tag_acc[26:][6 + 7],
                     ]  # df/hat_df multiver (df intersectional)
        tim_grp = [tag_sa1[3], tag_sa1[16], 'extGrp', 'extAlt']  # three
        tim_df = [tag_sa1[20], tag_sa1[27], tag_sa1[27 + 4], tag_sa1[
            27 + 4 + 7]]  # df4one sen-att: bin-val, multival, hat_df x2
        df_nonbin['idv_ptb'] = df_nonbin[tag_acc[-2]] + df_nonbin[
            tim_idv[-1]]  # 'idvDR_perturb','idv_dr_ptb', 'idvDR_'
        # self.draw_extended_grp_scat(df_nonbin, tag_sa1[:3] + [
        #     tag_sa1[16 + 3], tag_sa1[27 + 3]],
        #     tag_sa1[4:10][:3] + [tag_sa1[16 + 7], tag_sa1[27 + 7]],
        #     tag_sa1[10:16][:3] + [tag_sa1[16 + 10], tag_sa1[27 + 10]],
        #     f'{figname}_scat', verbose)
        # # pdb.set_trace()
        # self.draw_extended_grp_tim(df_nonbin, tag_sa1[3], [
        #     tag_sa1[16], 'extGrp', 'extAlt'], figname + '_grp', verbose)
        # self.draw_extended_hfm_tim(df_nonbin, tag_sa1[20], [
        #     tag_sa1[27], tag_sa1[27 + 4], tag_sa1[27 + 4 + 7]],
        #     figname + '_hfm')
        # self.draw_extended_idv_tim(df_nonbin, tim_grp[:2], [
        #     tim_idv, tim_df, tim_df_pl], figname + '_idv')

        if not verbose:
            return
        tag_grp = (tag_sa1[3], [tag_sa1[16], 'extGrp', 'extAlt'])
        tag_idv = (tim_grp[:2], [tim_idv, tim_df, tim_df_pl])
        tag_hfm = (tag_sa1[20], [
            tag_sa1[27], tag_sa1[27 + 4], tag_sa1[27 + 4 + 7]])
        self.avg_draw_extended_tim(df_nonbin, tag_grp, tag_idv, tag_hfm,
                                   figname + '_collect_tim', verbose)
        del tag_grp, tag_idv, tag_hfm
        return

    def schedule_mspaint_avg(self, raw_dframe, figname=''):
        mk, verbose = 'tst', False
        tag_acc, tag_sa1, _ = self.obtain_tag_col(mk)
        df_nonbin, tag_acc = self.data_gathering(*raw_dframe)
        pick = [0, 4, 5, 6]

        # extension in average forms, above is maximal forms
        col_grp = tag_sa1[:3] + [tag_sa1[16 + 3], tag_sa1[27 + 3]]
        col_ext = tag_sa1[4:10][-3:] + [tag_sa1[23], tag_sa1[34]]
        col_ext_alt = tag_sa1[10:16][-3:] + [tag_sa1[26], tag_sa1[37]]
        self.avg_draw_trade_off(df_nonbin, pick, tag_acc[
            :16] + [tag_acc[21], tag_acc[23], tag_acc[18], ], [
            col_grp, col_ext, col_ext_alt], f'{figname}_to_avg')
        if verbose:
            self.avg_draw_extended_grp_scat(
                df_nonbin, col_grp, col_ext + tag_sa1[4:7],
                col_ext_alt + tag_sa1[10:13], f'{figname}_scat_avg', verbose)

        col_ext_max = tag_sa1[4:10][:3] + [tag_sa1[23], tag_sa1[34]]
        col_ext_alt_max = tag_sa1[10:16][:3] + [
            tag_sa1[26], tag_sa1[37]]
        if verbose:
            self.avg_draw_trade_off_alt(
                df_nonbin, pick, tag_acc[:16] + [
                    tag_acc[21], tag_acc[23], tag_acc[18], ], [
                    col_grp, col_ext_max, col_ext_alt_max,
                    col_ext, col_ext_alt], f'{figname}_to_alt')
        self.avg_draw_incompatible_alt(
            df_nonbin, tag_acc[:16] + [
                tag_acc[21], tag_acc[23], tag_acc[18], ], [
                col_grp, col_ext_max, col_ext_alt_max,
                col_ext, col_ext_alt], f'{figname}_nc')

        if not verbose:
            return
        raw_df_b = raw_dframe[0]
        _, id_set = self.recap_sub_data(raw_df_b, sa_ir=3, sa_r=4)
        fgn = f'{figname}_radar_avg'
        self.avg_depict_separately(
            [2, 3, 4], [0, 1, 2, 6, 10], raw_df_b, id_set, mk, fgn)
        return


# -----------------------------


# -----------------------------
