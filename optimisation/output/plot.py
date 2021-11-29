import numpy as np
import pandas as pd
import string

import matplotlib as mpl
import matplotlib.style as style

import matplotlib.pyplot as plt
from lib.utils import truncate_colormap
# from matplotlib import rcParams

# TODO make this into a style file: https://matplotlib.org/stable/tutorials/introductory/customizing.html
# TODO convert otf files into ttf using this: https://pypi.org/project/otf2ttf/
# TODO some more font related stuff here: http://phyletica.org/matplotlib-fonts/

from matplotlib import font_manager
font_dirs = ['/Users/3s/Library/Fonts' ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Gulliver-Regular'

from matplotlib import cm
import seaborn as sns


def pareto_front(data, test_case, colour_by_generation=True, plot_final_generation_only=False, indices=None, feasible_only=True):

    # Plot settings
    # mpl.rc('lines', markersize=6)
    mpl.rc('axes', labelsize=12)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)
    mpl.rc('legend', fontsize=11)
    mpl.rc('figure', figsize=[6.4, 4.8])
    mpl.rc('savefig', dpi=200, format='pdata', bbox='tight')
    labels = list(string.ascii_uppercase)

    # Scaling objectives
    pd.options.mode.chained_assignment = None   # default='warn'
    f_obj_0_scale = -1.0
    f_obj_1_scale = -1.0
    data['f_0'] *= f_obj_0_scale
    data['f_1'] *= f_obj_1_scale
    ax_0_label = 'Weighted lift-to-drag'
    ax_1_label = 'Max. lift coefficient'

    # Generations
    generations = data['generation'].unique()

    # Plot pareto front
    if plot_final_generation_only:
        fig, ax = plt.subplots()

        gen_mask = np.in1d(data['generation'], generations[-1])
        gen_data = data[gen_mask]

        ax.scatter(gen_data['f_0'], gen_data['f_1'], facecolor='C0', alpha=0.5, label='Final population')
        ax.set_xlabel(ax_0_label)
        ax.set_ylabel(ax_1_label)
        ax.grid(True)
        ax.legend()
    else:
        if colour_by_generation:
            # Colormap
            n_gen = generations[-1]
            cmap = cm.get_cmap('viridis', n_gen)
            cmap = cm.get_cmap('Blues', n_gen)
            cmap = truncate_colormap(cmap, 0.2, 0.8)
            # line color
            myred = [216/ 255, 30/ 255, 49/ 255]
            myblue = [27/ 255, 99/ 255, 157/ 255]
            mygreen = [0, 128/ 255, 0]
            mycyan = [2/ 255, 169/ 255, 226/ 255]
            myyellow = [251/ 255, 194/ 255, 13/ 255]
            mygray = [89/ 255, 89/ 255, 89/ 255]
            mypurple = [0.4940, 0.1840, 0.5560]
            myaqua = [0.3010, 0.7450, 0.9330]

            mycolors = [[0, 0, 0], '#0F95D7', [213 / 255, 94 / 255, 0], [0, 114 / 255, 178 / 255],
                        [0, 158 / 255, 115 / 255], [230 / 255, 159 / 255, 0]]
            # style.use('fivethirtyeight')

            gen_mask = np.in1d(data['generation'], n_gen)
            non_dominated_mask = (data['rank'] == 0.0)
            mask = gen_mask & non_dominated_mask
            dominated_data = data[~mask]

            # Redoing things so that flipping the final generation dataframe generates the correct results
            gen_mask = np.in1d(data['generation'], n_gen)
            gen_data = data[gen_mask]
            gen_data = gen_data.iloc[::-1]
            non_dominated_mask = (gen_data['rank'] == 0.0)
            if feasible_only:
                non_dominated_data = gen_data[non_dominated_mask]
            else:
                non_dominated_data = gen_data

            fig, ax = plt.subplots()
            plt.scatter(dominated_data['f_0'].values, dominated_data['f_1'].values, c=dominated_data['generation'],
                        cmap=cmap, alpha=0.3, vmin=0)
            cbar = plt.colorbar(label='Generation', ticks=np.arange(0, n_gen+10, 10, dtype=np.int).tolist())
            cbar.ax.set_yticklabels(np.arange(0, n_gen+10, 10, dtype=np.int).tolist())
            cbar.set_alpha(1.0)
            cbar.draw_all()
            plt.scatter(non_dominated_data['f_0'].values, non_dominated_data['f_1'].values, color=myred, alpha=1,
                        label='Non-dominated solutions')
            label_indices(ax, indices, labels, non_dominated_data)
            ax.set_xlabel(ax_0_label)
            ax.set_ylabel(ax_1_label)
            ax.grid(True)
            plt.xlim(0, 100)
            plt.ylim(0, 2.0)
            # plt.ylim(0.4, 2.4)

        else:
            fig, ax = plt.subplots()

            for gen_idx in generations:
                gen_mask = np.in1d(data['generation'], gen_idx)
                gen_data = data[gen_mask]

                # Flipping dataframe so the worst ranked individuals are plotted first
                gen_data = gen_data.iloc[::-1]

                if gen_idx == generations[-1]:
                    non_dominated_mask = (gen_data['rank'] == 0.0)
                    if feasible_only:
                        non_dominated_data = gen_data[non_dominated_mask]
                    else:
                        non_dominated_data = gen_data
                    dominated_data = gen_data[~non_dominated_mask]
                    ax.scatter(dominated_data['f_0'], dominated_data['f_1'], facecolor='C0', alpha=0.3,
                               label='Dominated solutions')
                    ax.scatter(non_dominated_data['f_0'], non_dominated_data['f_1'], facecolor='C1', alpha=0.6,
                               label='Non-dominated solutions')
                    label_indices(ax, indices, labels, non_dominated_data)
                else:
                    ax.scatter(gen_data['f_0'], gen_data['f_1'], facecolor='C0', alpha=0.3)

            ax.set_xlabel(ax_0_label)
            ax.set_ylabel(ax_1_label)
            ax.grid(True)

    ax.set_xlim(30.0, 100.0)
    ax.set_ylim(1.2, 2.2)

    if np.sign(f_obj_0_scale) == -1:
        ax.invert_xaxis()
    if np.sign(f_obj_1_scale) == -1:
        ax.invert_yaxis()

    # Saving plot
    plt.savefig('../../figures/' + test_case + '_pareto_front' + '.pdf')

    plt.show()
    plt.close()


def label_indices(ax, indices, labels, non_dominated_data):

    if indices is not None:
        f_0 = non_dominated_data['f_0'].values
        f_1 = non_dominated_data['f_1'].values
        for i, idx in enumerate(indices):
            ax.annotate(labels[i],
                        xy=(
                        f_0[np.shape(non_dominated_data)[0] - 1 - idx], f_1[np.shape(non_dominated_data)[0] - 1 - idx]),
                        xytext=(-30, -15),
                        textcoords='offset points',
                        fontsize=11,
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


def scatterplot_matrix(df, test_case, n_gen, colour_by_generation=False, group_generations=False, plot_correlation=True):

    # Scale the column values for plotting
    pd.options.mode.chained_assignment = None  # default='warn'
    df['chord_control_pts_0'] *= 1000.0
    df['twist_control_pts_2'] *= (180.0/np.pi)
    df['twist_control_pts_3'] *= (180.0/np.pi)

    # Setting scaling factors for each objective
    f_obj_0_scale = 0.5
    f_obj_1_scale = 0.5

    # Scaling objectives
    df['f_0'] *= f_obj_0_scale
    df['f_1'] *= f_obj_1_scale

    # print('df.columns:', df.columns)
    # Rename the pandas columns here, so that seaborn will plot more suitable axis labels
    df = df.rename(columns={'radius_0': 'Radius [m]',
                            'chord_control_pts_0': 'Root chord [mm]',
                            'twist_control_pts_2': 'Twist at y = 0.667 [deg]',
                            'twist_control_pts_3': 'Tip twist [deg]',
                            'rank': 'Pareto Rank',
                            'generation': 'Generation'})

    # Rename case-specific columns
    df = df.rename(columns={'rpm_endurance_0': 'Endurance RPM [-]',
                            'rpm_hover_0': 'Hover RPM [-]',
                            'f_0': 'Endurance power frac. [-]',
                            'f_1': 'Hover power frac. [-]'})

    # Plot settings
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=14)
    mpl.rc('ytick', labelsize=14)
    mpl.rc('legend', fontsize=13)
    mpl.rc('savefig', dpi=100, format='pdf', bbox='tight')

    # Remove NaN Pareto ranks
    df = df[~np.isnan(df['Pareto Rank'])]

    # Boolean variable for plotting Rank 0 solutions from the final generation one colour, and everything else
    # another colour
    df['Pareto Dominant'] = pd.Series(np.zeros(df.shape[0], dtype=bool), index=df.index)

    # Setting Pareto Dominant flag for Rank 0 solutions from the final generation
    gen_mask = np.in1d(df['Generation'], n_gen - 1)
    non_dominated_mask = (df['Pareto Rank'] == 0.0)
    mask = gen_mask & non_dominated_mask
    df['Pareto Dominant'][mask] = True

    df['Generation'] = df['Generation'].astype(int)

    # Generation ranges
    if group_generations:
        df['Generation range'] = pd.Series(['']*df.shape[0], index=df.index)
        gen_range = 10
        generations = df['Generation'].unique()
        for generation in generations:
            gen_mask = np.in1d(df['Generation'], generation)
            lower = int(gen_range*np.floor(generation/gen_range))
            upper = lower + gen_range - 1
            df['Generation range'][gen_mask] = str(lower) + ' to ' + str(upper)

    # KDE plots only work for categories with > 1 data point
    if colour_by_generation:
        gen_var = 'Generation'
        if group_generations:
            gen_var = 'Generation range'
        generations = df[gen_var].unique()
        mask = np.zeros(df.shape[0], dtype=bool)
        for generation in generations:
            gen_mask = np.in1d(df[gen_var], generation)
            if np.count_nonzero(gen_mask) > 50:
                mask = (mask | gen_mask)
        # Removing generations with <= 1 data point
        df = df[mask]

    # Setting hue variable & colour palette
    hue_var = 'Pareto Dominant'
    palette = 'muted'
    if colour_by_generation:
        hue_var = 'Generation'
        palette = 'viridis'
        if group_generations:
            hue_var = 'Generation range'

    # Plotting zero ranks last
    hue_z_order = df[hue_var].unique()

    # Generate pairplot (do not include hue here if correlation is to be plotted)
    grid = sns.PairGrid(df, diag_sharey=False,
                        vars=['Radius [m]', 'Root chord [mm]', 'Twist at y = 0.667 [deg]', 'Endurance RPM [-]',
                              'Hover RPM [-]', 'Endurance power frac. [-]', 'Hover power frac. [-]'],
                        )   # hue=hue_var, palette=palette

    if plot_correlation:
        # Plot correlation coefficients on upper triangle (requires grid to have no hue property prior to this)
        grid.map_upper(corrdot)
    else:
        # Plot KDE on upper triangle
        grid.map_upper(sns.kdeplot)

    # Set grid hue & palette
    grid.hue_vals = df[hue_var]
    grid.hue_names = df[hue_var].unique()
    if not group_generations:
        grid.hue_kws = {'zorder': hue_z_order}
    grid.palette = sns.color_palette(palette, len(grid.hue_names))

    # Plot scatterplots on lower triangle
    grid.map_lower(plt.scatter, alpha=0.3)

    # Plot KDE on diagonal
    grid.map_diag(sns.kdeplot, lw=2, alpha=0.6, shade=False)

    # Set axes
    # grid.axes[0, 0].set_ylabel('Radius')    # Top left KDE plot is misleading if labelled using units
    # grid.axes[0, 0].set_xlim(0.125, 0.225)
    # grid.axes[0, 0].set_xticks(np.arange(0.15, 0.25, 0.05).tolist())
    # grid.axes[1, 1].set_xlim(0, 110)
    # grid.axes[1, 1].set_ylim(0, 110)
    # grid.axes[1, 1].set_xticks(np.arange(0, 150, 50).tolist())
    # grid.axes[1, 1].set_yticks(np.arange(0, 125, 25).tolist())
    # grid.axes[2, 2].set_xlim(-5, 60)
    # grid.axes[2, 2].set_ylim(-5, 60)
    # grid.axes[2, 2].set_xticks(np.arange(0, 60, 15).tolist())
    # grid.axes[2, 2].set_yticks(np.arange(0, 60, 15).tolist())
    # grid.axes[3, 3].set_xlim(3000, 14000)
    # grid.axes[3, 3].set_ylim(3000, 14000)
    # grid.axes[3, 3].set_xticks(np.arange(4000, 16000, 4000).tolist())
    # grid.axes[3, 3].set_yticks(np.arange(4000, 16000, 4000).tolist())
    # grid.axes[4, 4].set_xlim(4000, 16000)
    # grid.axes[4, 4].set_ylim(4000, 16000)
    # grid.axes[4, 4].set_xticks(np.arange(6000, 22000, 8000).tolist())
    # grid.axes[4, 4].set_yticks(np.arange(6000, 18000, 4000).tolist())
    # grid.axes[5, 5].set_xlim(0.0, 0.7)
    # grid.axes[5, 5].set_ylim(0.0, 0.7)
    # grid.axes[5, 5].set_xticks(np.arange(0.0, 1.0, 0.25).tolist())
    # grid.axes[5, 5].set_yticks(np.arange(0.0, 1.0, 0.25).tolist())
    # grid.axes[6, 6].set_xlim(0.3, 1.05)
    # grid.axes[6, 6].set_ylim(0.3, 1.05)
    # grid.axes[6, 6].set_xticks(np.arange(0.4, 1.3, 0.3).tolist())
    # grid.axes[6, 6].set_yticks(np.arange(0.4, 1.2, 0.2).tolist())

    # Add legend
    grid.add_legend()
    grid._legend.get_title().set_fontsize(13)

    # Save plot
    plt.savefig('../../figures/' + test_case + '_scatterplot_matrix' + '.pdf')

    plt.show()
    plt.close()


def corrdot(*args, **kwargs):

    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f'{corr_r:2.2f}'.replace('0.', '.')
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap='coolwarm',
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5],  xycoords='axes fraction',
                ha='center', va='center', fontsize=font_size)


def pareto_only(data, test_case, feasible_only=True):

    # Plot settings
    # mpl.rc('lines', markersize=6)
    mpl.rc('axes', labelsize=12)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)
    mpl.rc('legend', fontsize=11)
    mpl.rc('figure', figsize=[6.4, 4.8])
    mpl.rc('savefig', dpi=200, format='pdata', bbox='tight')
    labels = list(string.ascii_uppercase)

    # Scaling objectives
    pd.options.mode.chained_assignment = None   # default='warn'
    f_obj_0_scale = -1.0
    f_obj_1_scale = -1.0
    data['f_0'] *= f_obj_0_scale
    data['f_1'] *= f_obj_1_scale
    ax_0_label = 'Weighted lift-to-drag'
    ax_1_label = 'Max. lift coefficient'

    # Generations
    generations = data['generation'].unique()

    # Plot pareto front
    # Colormap
    n_gen = generations[-1]
    cmap = cm.get_cmap('viridis', n_gen)
    cmap = cm.get_cmap('Blues', n_gen)
    cmap = truncate_colormap(cmap, 0.2, 0.8)
    # line color
    myred = [216/ 255, 30/ 255, 49/ 255]
    myblue = [27/ 255, 99/ 255, 157/ 255]
    mygreen = [0, 128/ 255, 0]
    mycyan = [2/ 255, 169/ 255, 226/ 255]
    myyellow = [251/ 255, 194/ 255, 13/ 255]
    mygray = [89/ 255, 89/ 255, 89/ 255]
    mypurple = [0.4940, 0.1840, 0.5560]
    myaqua = [0.3010, 0.7450, 0.9330]

    mycolors = [[0, 0, 0], '#0F95D7', [213 / 255, 94 / 255, 0], [0, 114 / 255, 178 / 255],
                [0, 158 / 255, 115 / 255], [230 / 255, 159 / 255, 0]]
    # style.use('fivethirtyeight')

    gen_mask = np.in1d(data['generation'], n_gen)
    non_dominated_mask = (data['rank'] == 0.0)
    mask = gen_mask & non_dominated_mask
    dominated_data = data[~mask]

    # Redoing things so that flipping the final generation dataframe generates the correct results
    gen_mask = np.in1d(data['generation'], n_gen)
    gen_data = data[gen_mask]
    gen_data = gen_data.iloc[::-1]
    non_dominated_mask = (gen_data['rank'] == 0.0)
    if feasible_only:
        non_dominated_data = gen_data[non_dominated_mask]
    else:
        non_dominated_data = gen_data

    fig, ax = plt.subplots()
    plt.scatter(non_dominated_data['f_0'].values, non_dominated_data['f_1'].values, color=myred, alpha=1,
                label='Non-dominated solutions')
    ax.set_xlabel(ax_0_label)
    ax.set_ylabel(ax_1_label)
    ax.grid(True)
    plt.xlim(10, 60)
    plt.ylim(1.4, 2.4)
    # plt.ylim(0.4, 2.4)


    # ax.set_xlim(20.0, 70.0)
    # ax.set_ylim(0.0, 3.5)

    if np.sign(f_obj_0_scale) == -1:
        ax.invert_xaxis()
    if np.sign(f_obj_1_scale) == -1:
        ax.invert_yaxis()

    # Saving plot
    # plt.savefig('../../figures/' + test_case + '_pareto_front' + '.pdf')

    plt.show()
    plt.close()
