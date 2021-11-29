import os
import matplotlib.style as style


def plot_geometry(airfoil, plot_name='test'):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_dirs = ['/Users/3s/Library/Fonts']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)


    # Plot settings
    # mpl.rc('lines', linewidth=2, markersize=10)
    # mpl.rc('axes', labelsize=17)
    # mpl.rc('xtick', labelsize=17)
    # mpl.rc('ytick', labelsize=17)
    # mpl.rc('legend', fontsize=15)
    # mpl.rc('figure', figsize=[6.4, 4.8])
    # mpl.rc('savefig', dpi=300, format='pdf', bbox='tight')

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Gulliver-Regular'
    mycolors = [[0, 0, 0], '#0F95D7', [213 / 255, 94 / 255, 0], [0, 114 / 255, 178 / 255], [0, 158 / 255, 115 / 255],
              [230 / 255, 159 / 255, 0]]
    # style.use('fivethirtyeight')


    plt.figure()
    plt.plot(airfoil.x, airfoil.z, mycolors[1], alpha=0.75)
    plt.xlabel('x [--]')
    plt.ylabel('z [--]')
    plt.grid(b=True, which='major', linestyle='-', linewidth=0.25, alpha=0.25)
    plt.grid(b=True, which='minor', linestyle='--', linewidth=0.25, alpha=0.25)
    plt.axis('equal')

    figure_directory = '../../figures/'
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    plt.savefig(figure_directory + plot_name + '.pdf')

    plt.show()
