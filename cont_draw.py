# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from pyfair.facil.utils_const import DTY_FLT, subfig_ind
from pyfair.facil.draw_prelim import _style_set_axis, _setup_figshow
from pyfair.marble.draw_hypos import Pearson_correlation
from pyfair.granite.draw_graph import _sns_line_err_bars

from pyfair.granite.draw_fancy import _radar_X_revised
from pyfair.granite.draw_addtl import (
    _navy, _pl_myclr, _subproc_pl_lin_reg, _subproc_pl_lin_reg_alt,
    _subproc_pl_identity, _marginal_distr_read_in,
    _internal_marg_dist_s1, _internal_marg_dist_s2,
    _marginal_distr_step7a, _marginal_distr_step7b)


# ------------------------------
# draw_fancy.py


def _radar_sharey_ver1(fig, df_set, tag_Xs, annotX, annotY, stylish, sharey,
                       entitle):
    num_set, num_clf = len(df_set), len(df_set[0])
    outer = fig.add_gridspec(num_set, num_clf, wspace=.07, hspace=-1.42)  # .35, .64,.23
    if sharey:
        row_refs, col_refs = [None] * num_set, [None] * num_clf  # ref_ax=None
    # grid = ImageGrid(fig, 111, nrows_ncols=(num_set, num_clf),
    #                  axes_pad=.15, cbar_location="right", cbar_mode="single",
    #                  cbar_size="7%", cbar_pad=.15)
    for ik, ax_spec in enumerate(outer):
        ir, ic = ik // num_clf, ik % num_clf  # divmod(ik, num_clf)
        # grid[ir, ic].remove()
        # gs = GridSpecFromSubplotSpec(ir, ic, subplot_spec=fig.add_subplot(
        #     num_set, num_clf)[ir, ic], wspace=0.05, hspace=0.05)
        tk = {'sharex': col_refs[ic], 'sharey': row_refs[ir]} if sharey else {}
        inner = fig.add_subplot(ax_spec, projection='polar',  # outer[ir, ic],
                                **tk)  # sharex=ref_ax, sharey=ref_ax)
        if sharey:
            if row_refs[ir] is None:
                row_refs[ir] = inner
            if col_refs[ic] is None:
                col_refs[ic] = inner
            # if ic > 0:
            #     inner.set_yticklabels([])
            # if ir < num_set - 1:
            #     inner.set_xticklabels([])
            # if ref_ax is None:
            #     ref_ax = inner  # ax
            if ic != num_clf - 1:
                inner.set_yticklabels([])
                inner.set_yticks([])  # 彻底隐藏r轴刻度
            # 第一个子图作为共享参考， theta标签靠近圆边缘，r轴刻度 只在每一行的最后一列显示
            # inner.set_rorigin(-0.1)  # 轻微向内移动原点 让标签更靠近圆 # 让theta标签贴着圆圈
            inner.tick_params(pad=3.2)  # default:7
        inner = _radar_X_revised(inner, df_set[ir][ic], tag_Xs, annotX,
                                 clockwise=True, stylish=stylish)
        if entitle:
            inner.set_title(f'Panel ({chr(ik + 65).lower()}){"":31s}',
                            fontdict={'fontweight': plt.rcParams['axes.titleweight'],
                                      'color': 'dimgray'})  # 'darkgrey'})
        inner.set_xlabel(subfig_ind(ik), labelpad=-2)
        # for label, angle in zip(inner.get_xticklabels(), inner.get_xticks()):
        #     label.set_verticalalignment('center')
        #     label.set_horizontalalignment('center')
        #     label.set_y(inner.get_rmax().tolist())  # 把标签放到最大半径处
        # # 强制所有theta标签贴着圆圈
        if ic < num_clf - 1:
            continue
        plt.legend(annotY, bbox_to_anchor=(1.25, 1), frameon=False,
                   borderaxespad=0, loc='upper left', labelspacing=.07,
                   prop={'size': 9})
    return


def _radar_sharey_ver2(fig, df_set, tag_Xs, annotX, annotY, stylish, sharey,
                       entitle):
    num_set, num_clf = len(df_set), len(df_set[0])
    outer = fig.add_gridspec(num_set, num_clf, wspace=.07, hspace=.007)  # .35
    # 先创建所有子图
    axes = []
    for ik, ax_spec in enumerate(outer):
        ir, ic = divmod(ik, num_clf)
        inner = fig.add_subplot(ax_spec, projection='polar')
        axes.append((ir, ic, inner))
    # 计算全局r范围（如果你需要）
    rmax = max(ax.get_rmax() for _, _, ax in axes)
    # 第二遍：设置r范围+隐藏r标签
    for ir, ic, ax in axes:
        ax.set_rlim(0, rmax)  # 统一r范围，只在最后一列显示r
        if ic != num_clf - 1:
            ax.set_yticklabels([])
            ax.set_yticks([])
        ax.tick_params(pad=3.2)  # theta标签靠近圆

        ax = _radar_X_revised(ax, df_set[ir][ic], tag_Xs, annotX,
                              clockwise=True, stylish=stylish)
        if entitle:
            ax.set_title(f'Panel ({chr(ir * num_clf + ic + 65).lower()}){"":31s}',
                         fontdict={'fontweight': plt.rcParams['axes.titleweight'],
                                   'color': 'dimgray'})
        if ic < num_clf - 1:
            continue
        plt.legend(annotY, bbox_to_anchor=(1.25, 1), frameon=False,
                   borderaxespad=0, loc='upper left', labelspacing=.07,
                   prop={'size': 9})
    return


def radar_chart_gather(df_set, tag_Xs, annotX, annotY, figname='radar',
                       stylish=True, sharey=False,
                       entitle=True):  # clockwise=True,
    tt = len(annotX) // 2
    annotX[0] = "\n" + annotX[0]    # labels[0] = "\n" + labels[0]
    annotX[tt] = annotX[tt] + "\n"  # labels[4] = labels[4] + "\n"
    del tt  # pdb.set_trace()

    num_set, num_clf = len(df_set), len(df_set[0])
    from mpl_toolkits.axes_grid1 import ImageGrid
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    # fig = plt.figure(figsize=(13, 7.4), dpi=300, constrained_layout=True)
    # outer = fig.add_gridspec(num_set, num_clf)  # ,wspace=.64,hspace=.23) #.35

    fig = plt.figure(figsize=(14, 7.8), dpi=300, constrained_layout=True)
    _radar_sharey_ver1(fig, df_set, tag_Xs, annotX, annotY, stylish, sharey, entitle)
    # _radar_sharey_ver2(fig, df_set, tag_Xs, annotX, annotY, stylish, sharey, entitle)
    # plt.subplots_adjust(wspace=.34, hspace=0)  # ,**kw)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def _tabular_panel(grid, ir, ic, num_clf, data, algo, panel, pad=2):
    # grid[ir, ic].set_xticks([])
    # im = "Panel ({}) {:14s}".format(im, ) 
    im = "Panel ({})".format(chr(ir * num_clf + ic + 65).lower())
    if panel == 'y':
        grid[ir, ic].set_title(algo[ic] if ir == 0 else "", pad=pad)
        grid[ir, ic].set_ylabel("{}\n\n{}".format(data[ir] if ic == 0 else "", im))
    elif panel == 'x':
        grid[ir, ic].set_title(im + '  ' + (
            algo[ic] if ir == 0 else f"{'':16s}"), pad=pad)  # pad=-1)
        grid[ir, ic].set_ylabel(data[ir] + "\n" if ic == 0 else "")
    elif panel == 'z':
        grid[ir, ic].set_ylabel(data[ir] + "\n" if ic == 0 else "")
        grid[ir, ic].set_title(algo[ic] if ir == 0 else "")
        # grid[ir, ic].text(0.95, 0.95, im, color='dimgray', fontsize=10,
        #                   ha='right', va='top',
        #                   bbox=dict(facecolor='white', alpha=0.5))  # 可选：添加背景框
        # # 在右上角添加文字，(1,1)表示右上角。ha水平右对齐 va垂直顶对齐 确保文字不超出边框
        # grid[ir, ic].annotate(im, xy=(1, 1), arrowprops=dict(
        #     arrowstyle='->', connectionstyle='arc3'))
        # # ,facecolor='black',shrink=0.05),)  # xytest=(1,1),

    elif panel == 'yp':
        grid[ir, ic].set_title(algo[ic] if ir == 0 else "", pad=pad, fontsize=9)
        grid[ir, ic].set_ylabel((data[ir] if ic == 0 else "") + "\n\n", fontsize=9)  # im
        grid[ir, ic].set_xlabel(subfig_ind(ir * num_clf + ic), labelpad=pad + 47)  # 53.4)
        grid[ir, ic].tick_params(axis='y', labelsize=8)
    return  # return grid


def _tabular_sharey_ver1(grid, df_set, columns, rows, colors, cumulate, kws,
                         annotX=None):
    num_set, num_clf = len(df_set), len(df_set[0])
    if annotX is None:
        annotX = columns

    # row_refs, col_refs = [None] * num_set, [None] * num_clf
    for ir in range(num_set):
        for ic in range(num_clf):

            n_rows = df_set[ir][ic].shape[0]
            index = np.arange(len(columns)) + 0.3
            bar_width = 0.4
            y_offset = np.zeros(len(columns))
            cell_text = []
            for row in range(n_rows):
                # grid[ir, ic].bar(index, df_set[ir][ic].loc[row].values.astype(DTY_FLT), bar_width, bottom=y_offset, color=colors[row])
                # if cumulate:
                #     y_offset = y_offset + df_set[ir][ic].loc[row]
                # cell_text.append(['%1.4f' % x for x in df_set[ir][ic].loc[row]])

                vals = df_set[ir][ic].loc[row].values.astype(DTY_FLT)
                grid[ir, ic].bar(index, vals, bar_width, bottom=y_offset, color=colors[row])
                if cumulate:
                    y_offset = y_offset + vals                 # df_set[ir][ic].loc[row]
                cell_text.append(['%1.4f' % x for x in vals])  # df_set[ir][ic].loc[row]])
            cell_text.reverse()

            the_table = grid[ir, ic].table(
                cellText=cell_text, rowLabels=rows, rowColours=colors[::-1],
                colLabels=annotX, loc='bottom')  # colLabels=columns,
            # ax_table = inset_axes(grid[ir, ic], width="100%", height="30%",
            #                       loc='lower center', bbox_to_anchor=(0, -0.35, 1, 1),
            #                       bbox_transform=grid[ir, ic].transAxes, borderpad=0)
            # ax_table.axis('off')
            # ax_table.table(cellText=cell_text, rowLabels=rows, rowColours=colors[
            #     ::-1], colLabels=columns, loc='center')  # colLabels=annotX

            grid[ir, ic].set_xticks([])
            # grid[ir, ic].set_ylabel(data[ir])    # algo[ic])
            # grid[ir, ic].set_title(algo[ic])   # data[ir])

            # if row_refs[ir] is None:
            #     row_refs[ir] = grid[ir, ic]
            # if col_refs[ic] is None:
            #     col_refs[ic] = grid[ir, ic]
            # # if ic != num_clf - 1:
            # #     grid[ir, ic].set_yticklabels([])
            # #     grid[ir, ic].set_yticks([])
            _tabular_panel(grid, ir, ic, num_clf, *kws)
    # # plt.subplots_adjust(hspace=4.35, wspace=.35) #left=0.2,bottom=0.2)
    plt.subplots_adjust(left=0.2, bottom=0.2, hspace=4.35, wspace=.35)
    plt.tight_layout(rect=[0, 0, 1, 1], pad=.23)  # .47)  # pad=2.0)
    return


# def _tabular_sharey_ver2(grid, df_set, columns, rows, colors, cumulate, kws):
#     num_set, num_clf = len(df_set), len(df_set[0])
#     index, bar_width = np.arange(len(columns)) + 0.3, 0.4
#     from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#
#     row_refs, col_refs = [None] * num_set, [None] * num_clf
#     for ir in range(num_set):
#         for ic in range(num_clf):
#             ax = grid[ir, ic]
#             # 画柱状图
#             n_rows = df_set[ir][ic].shape[0]
#             y_offset = np.zeros(len(columns))
#             cell_text = []
#
#             for row in range(n_rows):
#                 vals = df_set[ir][ic].loc[row].values.astype(DTY_FLT)
#                 ax.bar(index, vals, bar_width, bottom=y_offset, color=colors[row])
#                 if cumulate:
#                     y_offset += vals
#                 cell_text.append([f"{x:1.4f}" for x in vals])
#             cell_text.reverse()
#
#             # 把table放到 Axes 外部（关键）
#             ax_table = inset_axes(
#                 ax, width="100%", height="28%", loc='lower center', bbox_to_anchor=(
#                     0, -0.32, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
#             ax_table.axis('off')
#             ax_table.table(cellText=cell_text, rowLabels=rows, rowColours=colors[
#                 ::-1], colLabels=columns, loc='center')
#             # ax_table.auto_set_font_size(False)  # tbl.
#             # ax_table.set_fontsize(7)            # tbl.
#
#             # 共享y轴（当前 现在完全正常）
#             if row_refs[ir] is None:
#                 row_refs[ir] = ax
#             else:
#                 ax.sharey(row_refs[ir])
#             # 隐藏x ticks
#             ax.set_xticks([])
#             # 设置标题和标签
#             _tabular_panel(grid, ir, ic, num_clf, *kws)
#     return


def tabular_chart_gather(df_set, tag_Xs, annotX, annotY, data='', algo='',
                         figname='tabular', cumulate=False, panel='y'):
    columns, rows = tag_Xs, annotY[::-1]
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    num_set, num_clf = len(data), len(algo)
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    fig, grid = plt.subplots(num_set, num_clf, constrained_layout=True,
                             figsize=(10.4, 5.7) if panel == 'yp' else (
                                 11.7, 6.4))  # (13,7.6),(14,7.8),(12,7.1)
    _tabular_sharey_ver1(  # _tabular_sharey_ver2(
        grid, df_set, columns, rows, colors, cumulate, [data, algo, panel],
        annotX)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# ------------------------------
# draw_addtl.py


def _rev_sup_pv1_wo_subtit(fig, df, existing, square_pm, *, share, lgth=3):
    pick, col_Xs, col_Y, tag_Yss, picked_keys = existing
    palette_X, palette_Y, antX, antYs, snspec, distrib = square_pm
    nk, ny = len(pick), len(antYs)  # wspace=.315, hspace=.25)

    grid = plt.GridSpec(ny * 5 + 1, nk * 5 + 1, wspace=.235, hspace=.25)
    row_refs, col_refs = [None] * ny, [None] * nk
    for ir in range(ny):
        for ic in range(nk):
            pk = pick[ic]
            dfs_pl, df_all = _marginal_distr_read_in(
                df, col_Xs[pk], col_Y, tag_Yss[ir], picked_keys)
            if distrib and ir == 0:
                ax00 = fig.add_subplot(grid[0, ic * 5: ic * 5 + 5])       # plt.subplot
                _internal_marg_dist_s1(ax00, df_all, col_Xs[pk], palette_X)
            if distrib and ic == nk - 1:
                ax11 = fig.add_subplot(grid[ir * 5 + 1: ir * 5 + 6, -1])  # plt.subplot
                _internal_marg_dist_s2(ax11, df_all, col_Y, palette_Y)
            # ax55 = plt.subplot(grid[ir * 5 + 1: ir * 5 + 6, ic * 5: ic * 5 + 5])

            sharey_ax, sharex_ax = row_refs[ir], col_refs[ic]
            kws = dict(sharex=sharex_ax, sharey=sharey_ax) if share else {}
            ax55 = fig.add_subplot(grid[ir * 5 + 1:ir * 5 + 6, ic * 5:ic * 5 + 5], **kws)
            if row_refs[ir] is None:
                row_refs[ir] = ax55
            if col_refs[ic] is None:
                col_refs[ic] = ax55
            if ic > 0:
                ax55.tick_params(labelleft=False)
            if ir < ny - 1:
                ax55.tick_params(labelbottom=False)

            ax55 = _marginal_distr_step7a(
                ax55, dfs_pl,  # picked_keys if gap else (..),
                # picked_keys[:3] if ir == 0 else picked_keys[-3:],
                picked_keys[:lgth] if ir == 0 else picked_keys[-lgth:],
                col_Xs[pk], col_Y, palette_Y,
                # ax55, dfs_pl, picked_keys, col_Xs[pk], col_Y, palette_Y,
                snspec, curr_key=ic == nk - 1, distrib=distrib)
            ax55 = _marginal_distr_step7b(
                ax55, antX[pk] if ir == ny - 1 else '',
                # subfig_ind(ir * nk + ic) + f"\n{antX[pk]}" * (ir == ny - 1),
                antYs[ir] if ic == 0 else '', distrib=distrib,
                _curr_lc=(1.11, .76))  # =(1.11, .86))

    del pick, col_Xs, col_Y, tag_Yss, picked_keys
    del palette_X, palette_Y, antX, antYs, snspec, distrib
    return fig


def _rev_sup_pv1_w_subtit(fig, df, existing, square_pm, *, share, lgth=3,
                          tit='bottom',  # ('top','bottom',False,'right'):#True,
                          start_pt_ind=0):
    pick, col_Xs, col_Y, tag_Yss, picked_keys = existing
    palette_X, palette_Y, antX, antYs, snspec, distrib = square_pm
    nk, ny = len(pick), len(antYs)  # wspace=.335, hspace=.35)

    ttp = 2 if tit in ['bottom', 'right'] else 3  # 2 + int(tit not in['b','r'])
    tt = 9  # 10
    grid = plt.GridSpec(ny * (tt + 1) + ttp, nk * tt + ttp,  # nk * tt + 2
                        wspace=.54 - .021 * (tit == 'bottom'), hspace=.31)
    # grid = plt.GridSpec(ny * 6 + 1, nk * 5 + 1, wspace=.135, hspace=.15)
    row_refs, col_refs = [None] * ny, [None] * nk
    for ir in range(ny):
        for ic in range(nk):
            pk = pick[ic]
            dfs_pl, df_all = _marginal_distr_read_in(
                df, col_Xs[pk], col_Y, tag_Yss[ir], picked_keys)
            tt_cc = ic * tt              # ic * tt + tt
            tt_rr = ir * (tt + 1) + ttp  # (ir + 1) * (tt + 1)

            if distrib and ir == 0 and start_pt_ind == 0:
                ax00 = fig.add_subplot(grid[0:2, tt_cc: tt_cc + tt])
                # ax00 = fig.add_subplot(grid[0, ic * 5: ic * 5 + 5])
                _internal_marg_dist_s1(ax00, df_all, col_Xs[pk], palette_X)
            if distrib and ic == nk - 1:
                ax11 = fig.add_subplot(grid[tt_rr:tt_rr + tt, -ttp:])  # -2:])
                # ax11 = fig.add_subplot(grid[ir * 6 + 1:ir * 6 + 6, -1])
                _internal_marg_dist_s2(ax11, df_all, col_Y, palette_Y)

            sharey_ax, sharex_ax = row_refs[ir], col_refs[ic]
            kws = dict(sharex=sharex_ax, sharey=sharey_ax) if share else {}
            ax55 = fig.add_subplot(grid[tt_rr:tt_rr + tt, tt_cc:tt_cc + tt], **kws)
            # ax55 = fig.add_subplot(grid[ir * 6 + 1: ir * 6 + 6, ic * 5:ic * 5 + 5], **kws)
            if not tit:
                ax55.tick_params(axis='x', which='major', pad=0)
                ax55.tick_params(axis='y', which='major', pad=1)
                ax55.xaxis.set_ticks_position('top')

            if row_refs[ir] is None:
                row_refs[ir] = ax55
            if col_refs[ic] is None:
                col_refs[ic] = ax55
            if ic > 0:
                ax55.tick_params(labelleft=False)
            if tit and ir < ny - 1:
                ax55.tick_params(labelbottom=False)
            if not tit and ir > 0:
                ax55.tick_params(labeltop=False)

            tt_curr = f"{antX[pk]}" * (ir == ny - 1)
            tt_curr = f"{tt_curr:8s}\n"  # :20s :8s
            tt_ind = subfig_ind(ir * nk + ic + start_pt_ind) + "\n"
            if tit == 'right':
                tt_curr = tt_ind.strip() + " " + tt_curr  # =tt_ind*(tit!='top')+tt_curr
            else:
                tt_curr = tt_ind * (not tit) + tt_curr + tt_ind.strip() * (tit == 'bottom')
            # tt_curr = (subfig_ind(ir * nk + ic) + "\n") * (tit not in [
            #     'top', 'bottom']) + tt_curr + "\n" + (
            #     subfig_ind(ir * nk + ic)) * (tit == 'bottom')  # tit!='top'
            ax55 = _marginal_distr_step7a(
                ax55, dfs_pl,
                picked_keys[:lgth] if ir == 0 else picked_keys[-lgth:],
                col_Xs[pk], col_Y, palette_Y,
                snspec, curr_key=ic == nk - 1, distrib=distrib)
            ax55 = _marginal_distr_step7b(
                ax55, tt_curr.strip(), antYs[ir] * (ic == 0),
                distrib=distrib, _curr_lc=(1.11, .76), pad=2 * (
                    not tit) + 3 * (tit == 'top') + 1 * (tit == 'bottom'),
                handles=None if ic == nk - 1 else [])
            if tit == 'top':
                ax55.set_title(subfig_ind(ir * nk + ic), pad=3, fontsize=9)
            ax55.tick_params(axis='both', labelsize=7.8, pad=.6)  # labelsize=7)

    del tt_cc, tt_rr, tt, ttp, tt_curr, tt_ind
    del pick, col_Xs, col_Y, tag_Yss, picked_keys
    del palette_X, palette_Y, antX, antYs, snspec, distrib
    return fig


# Revision


def _subtim_afterbody(ax, annots, sci_format_y, _curr_ft):
    ax.set_xlabel(annots[0], fontsize=9, family=_curr_ft, x=.55)
    ax.set_ylabel(annots[1], fontsize=9, family=_curr_ft, y=.55)
    if sci_format_y:
        ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
        ax.yaxis.get_offset_text().set_fontsize(8)
    return ax  # tail


def _subtim_sing_lin(ax, X, Y, snspec='sty2',
                     # lbl_X='x', lbl_Y='y', lbl_Z=r'$f(x)=x$'):
                     annots=('X', 'Y', 'Z'), sci_format_y=False):
    R = Pearson_correlation(X, Y)[0]
    # R = np.corrcoef(X, Y)[1, 0]
    key = 'Correlation = %.4f' % R
    regr = np.polyfit(X, Y, deg=1)
    estimated = np.polyval(regr, X)
    Z = sorted(X)
    annotZ = annots[2] if len(annots) > 2 else r'$f(x)=x$'

    kw = dict(alpha=1, linewidths=.4,)
    if snspec in ['sty6', 'sty9']:
        tx = _navy if snspec == 'sty6' else _pl_myclr[1]
        ax.scatter(x=X, y=Y, edgecolors='w', facecolor=tx, linewidths=.4)
        tx_min, tx_max = ax.get_xlim()
        ax.plot([tx_min, tx_max], [0, 0], 'k--', lw=1, label=annotZ)
        del tx, tx_max, tx_min
    # if snspec == 'sty2':
    #     ax.scatter(edgecolors='w', linewidths=.5, label=key, **kw)
    #     ax.plot(X, estimated, 'k-', lw=1)
    # elif snspec == 'sty3a':
    #     ax.scatter(edgecolors='w', linewidths=.4, facecolor=_navy, **kw)
    #     ax.plot(X, estimated, '-', lw=1, label=key, color=_navy)
    #     ax.plot(Z, Z, 'k--', lw=1, label=lbl_Z)
    _style_set_axis(ax)

    _curr_ft = plt.rcParams['font.family']
    legend_font = {'size': 8.7, 'family': _curr_ft}
    if snspec not in ['sty6', 'sty9']:
        kw = {'color': _navy, 'lw': 1}
        _sns_line_err_bars(ax, kw, X, Y)
    ax.legend(prop=legend_font, loc='best', frameon=False)
    ax = _subtim_afterbody(ax, annots, sci_format_y, _curr_ft)
    ax.tick_params(width=.6, length=2.5, labelsize=8.7)
    return ax


def _subtim_multi_lin(ax, X, Ys, snspec, annots, Zs,
                      sci_format_y=False):
    ax.tick_params(width=.6, length=2.5, labelsize=8.7)
    _curr_ft = plt.rcParams['font.family']
    legend_font = {'family': _curr_ft, 'size': 8.7}

    myclr = _pl_myclr
    if snspec.startswith('sty7'):
        myclr = [_navy, ] + myclr[:6]
        del myclr[4]
        del myclr[2:4]
        snspec = snspec.replace('y7', 'y6')
    annotZ = annots[2] if len(annots) > 2 else r'$f(x)=x$'

    n_k = len(Ys)
    start_i = 2 if n_k == 2 else 1
    for i in range(n_k):
        _subproc_pl_lin_reg(
            ax, X, Ys[i], Zs[i], annotZ, snspec, myclr[i + start_i])
    for i in range(n_k):
        _subproc_pl_lin_reg_alt(ax, X, Ys[i], snspec, myclr[i + start_i])
    _subproc_pl_identity(ax, [X, X], annotZ, snspec)

    if snspec.startswith('sty6'):
        _curr_fram = {'frameon': True, 'framealpha': .5, 'loc': 'best'}
    ax.legend(prop=legend_font, labelspacing=.14, handletextpad=.21, **_curr_fram)
    ax = _subtim_afterbody(ax, annots, sci_format_y, _curr_ft)
    _style_set_axis(ax)
    return ax


def linreg_w_marg_dist_rev_sup_pv1(
        df, col_Y, pick, col_Xs, tag_Yss, picked_keys, figname='smd',
        snspec='sty4b', antYs=('fair',), antX=('acc',), distrib=True,
        palette_X=('#F0AE97',) * 3,  # gap=False, #curr_key=False,invt_a=False,
        palette_Y=('#168E6A', ) + ('#1E827B', '#586395') * 4,
        subfig=False, start_pt_ind=0):
    # from mpl_toolkits.axes_grid1 import ImageGrid
    # fig = plt.figure(figsize=(15, 10), dpi=300)
    # grid = ImageGrid(fig, 111, nrows_ncols=(len(antYs), nk),
    #                  axes_pad=.15, cbar_location='right',
    #                  cbar_mode='single', cbar_size='7%', cbar_pad=.15)
    # for ik, ax in enumerate(grid):
    #     ir, ic = ik // nk, ik % nk

    fig = plt.figure(figsize=(7.8, 4.27 if subfig else 4.01), dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)

    # '' '
    # nk, ny = len(pick), len(antYs)
    # from matplotlib.gridspec import GridSpecFromSubplotSpec
    # fig, grid = plt.subplots(ny, nk, figsize=(11, 6.8), constrained_layout=True)
    # # fig = plt.figure(figsize=(10, 8), dpi=300)
    # # outer = fig.add_gridspec(ny, nk)
    # # 用来存储每行/每列的参考轴 (主轴)
    # row_refs = [None] * ny  # 每一行一个 y参考
    # col_refs = [None] * nk  # 每一列一个 x参考
    # for ir in range(ny):
    #     for ic in range(nk):
    #         dfs_pl, df_all = _marginal_distr_read_in(
    #             df, col_Xs[pick[ic]], col_Y, tag_Yss[ir], picked_keys)
    #
    #         grid[ir, ic].remove()  # 1. 先删除原来的占位子图
    #         curr_r, curr_c = 6 - int(ir != 0), 6 - int(ic != nk - 1)
    #         # 2. 在原位置创建一个新的 GridSpec
    #         gs = GridSpecFromSubplotSpec(curr_r, curr_c,
    #                                      subplot_spec=fig.add_gridspec(ny, nk)[
    #                                          ir, ic], wspace=0.05, hspace=0.05)
    #         # inner = outer[ir, ic].subgridspec(6, 6, wspace=.05, hspace=.05)
    #         # 3. 在这个 6x6 网格中创建子图
    #         # ax00 = fig.add_subplot(inner[0, 0:5])  # gs[0, 0])
    #         # ax11 = fig.add_subplot(inner[1:6, 5])  # gs[1:3, 1:4])
    #         # ax55 = fig.add_subplot()  # gs[5, :])  # ax55.axis('off')
    #         # 你可以继续在 gs[...] 中添加更多子图
    #
    #         # 这里我们约定：每个cell里有一个主轴，用来参与sharex/sharey，比如就用右下角那个
    #         main_r, main_c = curr_r - 1, curr_c - 1
    #         # 找到这一行/列的参考轴（如果已有的话）
    #         sharey_ax, sharex_ax = row_refs[ir], col_refs[ic]
    #
    #         if distrib and ir == 0:
    #             ax00 = fig.add_subplot(gs[0, 0:5])  # gs[0, 0:5])
    #             _internal_marg_dist_s1(ax00, df_all, col_Xs[pick[ic]], palette_Y)
    #         if distrib and ic == nk - 1:
    #             ax11 = fig.add_subplot(gs[int(ir == 0):, 5])  # gs[1:6, 5])
    #             _internal_marg_dist_s2(ax11, df_all, col_Y, palette_Y)
    #         ax55 = fig.add_subplot(gs[int(ir == 0):, 0:5],  # gs[1:6, 0:5])
    #                                sharex=sharex_ax, sharey=sharey_ax)
    #
    #         # 如果这一行还没有y参考轴，就把当前主轴记为参考
    #         # 如果这一列还没有x参考轴，就把当前主轴记为参考
    #         if row_refs[ir] is None:
    #             row_refs[ir] = ax55  # ax_main
    #         if col_refs[ic] is None:
    #             col_refs[ic] = ax55  # ax_main
    #         # 关键：隐藏重复 tick labels
    #         if ic > 0:
    #             ax55.tick_params(labelleft=False)
    #         if ir < ny - 1:
    #             ax55.tick_params(labelbottom=False)
    #
    #         ax55 = _marginal_distr_step7a(
    #             ax55, dfs_pl, picked_keys, col_Xs[pick[ic]], col_Y,
    #             palette_Y, snspec, curr_key=ic == nk - 1, distrib=distrib)
    #         ax55 = _marginal_distr_step7b(
    #             ax55, antX[pick[ic]] if ir == ny - 1 else '',
    #             antYs[ir] if ic == 0 else '', distrib=distrib,
    #             _curr_lc=(.51, .81))
    #     # if ir == 0:
    #     #     _marginal_distrib_step1(grid, df_all=, colx)
    # '' '

    existing = [pick, col_Xs, col_Y, tag_Yss, picked_keys]
    sqr_pm = [palette_X, palette_Y[:3], antX, antYs, snspec, distrib]
    if not subfig:
        fig = _rev_sup_pv1_wo_subtit(fig, df, existing, sqr_pm, share=True)
    else:
        fig = _rev_sup_pv1_w_subtit(fig, df, existing, sqr_pm, share=True,
                                    start_pt_ind=start_pt_ind)
    # pdb.set_trace()

    _setup_figshow(fig, figname=figname)
    return


def linreg_w_marg_dist_rev_sup_pv2(
        df, col_Y, pick, col_Xs, tag_Yss, picked_keys, figname='smd',
        snspec='sty4b', antYs=('fair',), antX=('acc',), distrib=True,
        palette_X=('#00a087',) * 5, palette_Y=(  # gap=True,
            'black', '#066190', '#C42238', '#024163', '#8E0F31',
            '#77AECD', '#D98380', '#066190', '#C42238'),
        subfig=True, start_pt_ind=0):  # starting_point_index
    nk, ny = len(pick), len(antYs)
    fig = plt.figure(figsize=(9.9, 6.7 if subfig else 5.76), dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    ttd, tt, ttp = (5, 3, 1) if not subfig else (10, 6, 3)  # (7, 4, 2)
    grid = plt.GridSpec(ny * (ttd + int(subfig)) + 1 + 1, nk * (ttd + tt),
                        wspace=.364 - .1 * subfig, hspace=.3 - .1 * subfig
                        )  # wspace=.364,hspace=.32)
    row_refs, col_refs = [None] * ny, [None] * nk
    for ir in range(ny):
        for ic in range(nk):
            pk = pick[ic]
            dfs_pl, df_all = _marginal_distr_read_in(
                df, col_Xs[pk], col_Y, tag_Yss[ir], picked_keys)
            tt_cc = ic * (ttd + tt)                 # ic * 8: ic * 8 + 5
            tt_rr = ir * (ttd + int(subfig)) + ttp  # ir * 5 + 1: ir * 5 + 6

            if distrib and ir == 0 and start_pt_ind == 0:
                ax00 = plt.subplot(grid[:ttp, tt_cc: tt_cc + ttd])
                # ax00 = plt.subplot(grid[0, ic * 8: ic * 8 + 5])
                _internal_marg_dist_s1(ax00, df_all, col_Xs[pk], palette_X)
            if distrib and ic == nk - 1:
                ax11 = plt.subplot(grid[tt_rr: tt_rr + ttd, -tt:-tt + ttp])
                # ax11 = plt.subplot(grid[ir * 5 + 1: ir * 5 + 6, -3])
                _internal_marg_dist_s2(ax11, df_all, col_Y, palette_Y)
            # ax55 = plt.subplot(grid[ir * 5 + 1:ir * 5 + 6, ic * 8:ic * 8 + 5])
            sharey_ax, sharex_ax = row_refs[ir], col_refs[ic]
            ax55 = fig.add_subplot(
                grid[tt_rr:tt_rr + ttd, tt_cc:tt_cc + ttd],
                # grid[ir * 5 + 1:ir * 5 + 6, ic * 8:ic * 8 + 5],
                sharex=sharex_ax, sharey=sharey_ax)
            if row_refs[ir] is None:
                row_refs[ir] = ax55
            if col_refs[ic] is None:
                col_refs[ic] = ax55
            if ic > 0:
                ax55.tick_params(labelleft=False)
            if ir < ny - 1:
                ax55.tick_params(labelbottom=False)

            tt_curr = f"\n{antX[pk]}" * (ir == ny - 1)
            tt_curr = tt_curr + ("\n" + subfig_ind(
                ir * nk + ic + start_pt_ind)) * subfig
            ax55 = _marginal_distr_step7a(
                ax55, dfs_pl, picked_keys, col_Xs[pk], col_Y, palette_Y,
                snspec, curr_key=ic == nk - 1, distrib=distrib)
            ax55 = _marginal_distr_step7b(
                ax55, tt_curr.strip(),  # antX[pk] if ir == ny - 1 else '',
                # subfig_ind(ir * nk + ic) + f"\n{antX[pk]}" * (ir == ny - 1),
                antYs[ir] if ic == 0 else '', distrib=distrib,
                # _curr_lc=(1.04,.61), (1.11,.76))
                _curr_lc =((1.09 if ic == nk - 1 else .98) - 0.012 * subfig,
                           0.56 + 0.02 * subfig), pad=4 * (not subfig))
            if subfig:
                ax55.tick_params(axis='both', pad=.6)
    del tt_curr, tt, ttd, tt_cc, tt_rr
    _setup_figshow(fig, figname=figname)
    return


def gathering_lin_reg_sup_tim(df, X, Ys, figname, ant_X, ant_Ys, lbl_Zs):
    num = len(X)     # figsize=(8.7-10.7, 4.27-4.37)
    # fig, ax = plt.subplots(nrows=2, ncols=num, sharex='col',  # True,
    #                        figsize=(9.86, 4.17), dpi=300,
    #                        constrained_layout=True)
    # # plt.subplots_adjust(left=.06, bottom=.06, right=.98, top=.995)

    import matplotlib.gridspec as gridspec  # 设置较大的画布   (11.6, 4.37)
    fig = plt.figure(figsize=(12.1, 4.42), dpi=300)  # ,constrained_layout=True)
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1.15, 1.09, 1.12],
                           figure=fig, wspace=.3, hspace=.2)  # gs[0, i]
    # 3. 定义 GridSpec：1行3列，中间列比两边宽 (宽比例为 1:2:1)
    # gs = gridspec.GridSpec(1, 3, width_ratios=[1, 2, 1])
    # axes = []
    # for r in range(2):
    #     tmp = []
    #     for c in range(4):
    #         ax = fig.add_subplot(gs[r, c])
    #         tmp.append(ax)
    #     axes.append(tmp)  # ax)
    # ax = axes
    axes = [fig.add_subplot(gs[0, c]) for c in range(4)]
    for ax in axes:   # 第一行
        ax.tick_params(labelbottom=False)
    tmp = [fig.add_subplot(gs[1, c], sharex=axes[c]) for c in range(4)]
    ax = [axes, tmp]
    del tmp, axes  # pdb.set_trace()

    for i in [0, 2]:   # if i % 2 == 0:
        tag_X, tag_Y = X[i], Ys[i]
        num_X = df[tag_X].values.astype(DTY_FLT)
        num_Y = df[tag_Y].values.astype(DTY_FLT)
        num_Y = num_Y / num_X
        snspec = 'sty6' if i == 0 else 'sty9'
        curr_antYs = ant_Ys[i] + r'$-1$'
        annots = [subfig_ind(i), curr_antYs, lbl_Zs[i]]
        ax[0][i] = _subtim_sing_lin(ax[0][i], num_X, num_Y - 1, snspec, annots)
        annots[0] = ant_X[i] + '\n' + subfig_ind(num + i)
        annots[1] = r'$\lg($' + ant_Ys[i] + r'$)$'
        ax[1][i] = _subtim_sing_lin(ax[1][i], num_X, np.log10(num_Y), snspec, annots)
        del tag_X, tag_Y, num_X, num_Y, snspec, curr_antYs, annots

    for i in [1, 3]:   # for i, tag_X in enumerate(X):
        tag_Ys = Ys[i]  # cur_antX,cur_antYs = ant_X[i],ant_Ys[i]
        num_X = df[X[i]].values.astype(DTY_FLT)
        num_Ys = [df[k].values.astype(DTY_FLT) / num_X - 1 for k in tag_Ys]
        snspec = 'sty7' if i == 1 else 'sty6'
        j = i if i <= 2 else i - 1
        curr_antYs = ant_Ys[j] + r'$-1$'
        annots = [subfig_ind(i), curr_antYs, lbl_Zs[i][0]]
        ax[0][i] = _subtim_multi_lin(ax[
            0][i], num_X, num_Ys, snspec, annots, lbl_Zs[i][1:])
        annots[0] = ant_X[j] + '\n' + subfig_ind(num + i)
        annots[1] = r'$\lg($' + ant_Ys[j] + r'$)$'
        ax[1][i] = _subtim_multi_lin(ax[1][i], num_X, [np.log10(
            k + 1) for k in num_Ys], snspec, annots, lbl_Zs[i][1:])
        # if i == 1:
        #     break
        del tag_Ys, num_X, num_Ys, snspec, j, curr_antYs, annots

    # fig.set_constrained_layout_pads(
    #     w_pad=4.0,   # 子图之间的水平间距
    #     h_pad=2.0,   # 子图之间的垂直间距
    #     wspace=0.2, hspace=0.2)  # 轴之间的额外空间
    # plt.tight_layout()  # pdb.set_trace()
    # fig.subplots_adjust(wspace=16, hspace=14)
    _setup_figshow(fig, figname=figname)
    return


# ------------------------------
