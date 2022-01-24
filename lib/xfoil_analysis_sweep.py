import numpy as np
import os, subprocess

from lib.utils import setup_solver
from lib import config, util
from lib.airfoil_parametrisation import AirfoilGeometry
from lib import flight
# from lib.result import Result2 as Result
from lib.design import Design
from lib.cfd_lib import gmsh_util, construct2d_util, xfoil_util, mses_util
from cases.single_element_setup import SingleElementSetup
from lib.graphics import plot_geometry


import matplotlib as mpl
import matplotlib.pyplot as plt

class Nexus():
    def __init__(self):
        self.fileinfo = FileInfo()
        self.input = Input()
        self.results = Results()

class FileInfo():
    def __init__(self):
        self.xfoil_PATH = '/usr/local/bin/'
        self.foil_name = ''
        self.savename = ''
        self.code = 1111
        self.number = 0
        self.x = None
        self.z = None
        self.difficult = False
        self.reynolds_run = False
    def set(self, foil_name, code, number):
        self.foil_name = foil_name
        self.code = code
        self.number = number
    def setcoords(self, x, z):
        self.x = x
        self.z = z

class Input():
    def __init__(self):
        self.mach = 0.2
        self.reynolds = 1e6
        self.alpha = []
        self.cl = []
        self.cl_ref = 0.1
        self.identifier = 0
    def set(self, mach, reynolds, alpha, cl, cl_ref, identifier):
        self.mach = mach
        self.reynolds = reynolds
        self.alpha = alpha
        self.cl = cl
        self.cl_ref = cl_ref
        self.identifier = identifier

class Results():
    def __init__(self):
        self.alpha = []
        self.cl = []
        self.cd = []
        self.cm = []
        self.info = {}

def write_input(fileinfo, input):
    name = fileinfo.foil_name
    input_name = '../'+name + '.inp'
    with open(input_name, 'w') as f:
        f.write('load ../airfoils/' + name + '.txt\n')
        f.write('oper\n')
        # if fileinfo.difficult:
        #     f.write('iter 50\n')
        f.write('A 0\n')
        f.write('Type 2\n')
        f.write('visc ' + str(int(input.reynolds))+'\n')
        f.write('m ' + str(input.mach)+'\n')
        f.write('C ' + str(input.cl[0]) + '\n')
        f.write('!\n')
        if not fileinfo.difficult:
            f.write('pacc\n')
            f.write('../' + name + '.pol\n\n')
            f.write('cseq {:1.2f} {:1.2f} {:1.2f}\n'.format(input.cl[0], input.cl[1], input.cl[2]))
            f.write('pacc')
        else:
            intervals = np.around(np.linspace(cl[0],cl[1],10),2)
            delta   = 0.001 #
            for i in range(0,len(intervals)-1):
                f.write('c {:1.2f}\n!\n'.format(intervals[i]))
                f.write('pacc\n')
                f.write('../' + name + '.pol\n\n')
                f.write('cseq {:1.2f} {:1.2f} {:1.3f}\n'.format(intervals[i], intervals[i+1], delta))
                if i == len(intervals)-2:
                    f.write('pacc')
                else:
                    f.write('pacc\n')
        f.close()

def run_xfoil(fileinfo, input):
    name = fileinfo.foil_name
    input_file = '../' + name + '.inp'
    log_name = '../' + name + '.log'
    if os.path.isfile('../' + name + '.pol'):
        os.remove('../' + name + '.pol')
    subprocess.run([fileinfo.xfoil_PATH + 'xfoil' + ' < ' + input_file + ' > ' + log_name ], shell=True)#, timeout=30)
    os.remove(input_file)
    os.remove(log_name)

def read_xfoil(fileinfo, input):
    results = Results()
    polar_file = '../' + fileinfo.foil_name + '.pol'
    # if os.path.isfile(polar_file):
    data = np.loadtxt(polar_file, skiprows=12)
    alpha = []
    cl = []
    cd = []
    try:
        for ct in range(0,len(data[:,0])):
            alpha.append(data[ct,0])
            cl.append(data[ct,1])
            cd.append(data[ct,2])
    except Exception as e:
        alpha = None
        cl = None
        cd = None

    results.alpha = alpha
    results.cl = cl
    results.cd = cd
    return results

def write_results_to_file(fileinfo,results):
    string = fileinfo.foil_name
    filename = '../'+fileinfo.savename+'_results'# string + '_results'
    with open(filename, 'w') as f:
        f.write('%s\n' % string)
        if type(results.alpha) == type(list()):
            for i in range(len(results.alpha)-1):
                f.write('%s %s %s\n' % (results.alpha[i], results.cl[i], results.cd[i]))
            f.write('%s %s %s' % (results.alpha[-1], results.cl[-1], results.cd[-1]))
        else:
            f.write('%s %s %s\n' % (0,0,0))
    f.close()

def read_results_to_nexus(nexus):
    results = Results()
    polar_file = '../' +nexus.fileinfo.savename+'_results'#+ nexus.fileinfo.foil_name + '_results'
    # with open(filename,'r') as f:
    data = np.loadtxt(polar_file, skiprows=1)
    try:
        results.alpha = data[:,0]
        results.cl = data[:,1]
        results.cd = data[:,2]
    except Exception as e:
        results.alpha = None
        results.cl = None
        results.cd = None
    return results

def plot_robust_comparison(runs, choice, savename = 'xfoil_sweep_results.png', plot_type = 1):#
    # colors = [myblack, myblue, myred, myyellow, mygreen, mybrown, mydarkblue, mypurple, myorange, mygray]
    # choice = [foil names list]
    # plot_type: 1 - ld v cl, 2 cl v alpha - for plot 3
    sel = {} # selection
    for i, ch in enumerate(runs):
        if ch.fileinfo.foil_name in choice:
            if type(ch.results.cl) == type(None):
                print(ch.fileinfo.foil_name + ' is a FAILED RUN and will not be plotted')
            else:
                sel[i] = ch
        elif ch.fileinfo.savename in choice:
            if type(ch.results.cl) == type(None):
                print(ch.fileinfo.foil_name + ' is a FAILED RUN and will not be plotted')
            else:
                sel[i] = ch

    for key in sel.keys():
        kk = int(key)
        data= np.loadtxt('../airfoils/' + sel[key].fileinfo.foil_name + '.txt', skiprows=1)
        sel[key].fileinfo.setcoords(data[:, 0], data[:, 1])
    # plot this shit
    mycols = ['firebrick','darksalmon','sienna','goldenrod','olive','limegreen','seagreen','teal','darkturquoise','dodgerblue','slategrey','blueviolet','purple','deeppink']
    import random
    cc = random.sample(range(0,len(mycols)), len(list(sel.keys())))
    cc = [0,2,4,6,-2]
    linestyle = ['-','--','-.',':','-','--','-.',':','-','--','-.',':','-','--','-.',':']
    ll = linestyle[0:len(choice)]
    # fig1, (ax1,ax2,ax3) = plt.subplots(3)
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(9,7.5))
    gs = gridspec.GridSpec(2,2)
    ax1 = plt.subplot(gs[0,:])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[1,1])
    # ax1.title.set_text(savename)
    ax1.grid(color = 'gainsboro')
    ax2.grid(color = 'gainsboro')
    ax3.grid(color='gainsboro')
    # fig2, ax2 = plt.subplots(1)
    for i,k in enumerate(sel.keys()):
        if sel[0].fileinfo.reynolds_run:
            label = 'Mach: {:.2f}, Re: {:.1e}'.format(sel[k].input.mach, sel[k].input.reynolds)
        else:
            # label = sel[k].fileinfo.foil_name
            if sel[k].fileinfo.foil_name[0:9] in ['utest_unc']:
                label = 'Robust'
            else:
                label = 'Determinate'

        ax1.plot(sel[k].fileinfo.x,sel[k].fileinfo.z,mycols[cc[i]],linestyle=ll[i], label=label)
        ax2.plot(sel[k].results.cd,sel[k].results.cl,mycols[cc[i]],linestyle=ll[i])#,label = sel[k].fileinfo.foil_name)]
        if plot_type == 1:
            cl_cd = [sel[k].results.cl[j]/sel[k].results.cd[j] for j in range(0,len(sel[k].results.cl))]
            # ax3.plot(cl_cd, sel[k].results.cl, mycols[cc[i]],linestyle=ll[i])
            ax3.plot(sel[k].results.cl, cl_cd, mycols[cc[i]],linestyle=ll[i])
        elif plot_type == 2:
            ax3.plot(sel[k].results.alpha, sel[k].results.cl, mycols[cc[i]],linestyle=ll[i])
    ax1.axis('equal')
    ax2.legend(loc = 'lower right')
    ax2.set_xlabel(r'$C_d$',fontsize=14)
    ax2.set_ylabel(r'$C_l$',fontsize=14)
    ax2.set_ylim([0.0, 1.5])
    ax2.set_xlim([0.005, 0.045])
    ax3.set_ylim([10, 120])
    ax3.set_xlim([0.0, 1.5])
    if plot_type == 1:
        ax3.set_ylabel(r'$L/D$',fontsize=14)
        ax3.yaxis.set_ticks_position('right')
        ax3.yaxis.set_label_position('right')
        ax3.set_xlabel(r'$C_l$', fontsize=14)
    elif plot_type == 2:
        ax3.set_xlabel(r'$\alpha$',fontsize=14)
    # ax3.set_ylabel(r'$C_l$')
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0+box.height*0.2, box.width, box.height*0.9])
    # ax1.legend(loc='center left',fontsize=8, bbox_to_anchor=(0,-1),ncol=int(len(list(sel.keys()))/2))
    ncols = 3#len(list(sel.keys()))#3#min(4,int(len(list(sel.keys()))/2))
    ax1.legend(loc='upper center',bbox_to_anchor=(0.5,-0.125),fontsize=12,ncol=ncols)
    # plt.subplots_adjust(right=0.7)
    # plt.tight_layout()
    plt.savefig('../' + savename + '.png',dpi=250)
    plt.show()

def plot_robust_comparison2(runs, savename = 'xfoil_sweep_results.png', plot_type = 1):#
    # colors = [myblack, myblue, myred, myyellow, mygreen, mybrown, mydarkblue, mypurple, myorange, mygray]
    # choice = [foil names list]
    # plot_type: 1 - ld v cl, 2 cl v alpha - for plot 3
    sel = {} # selection
    for i, ch in enumerate(runs):
        if type(ch.results.cl) == type(None):
            print(ch.fileinfo.foil_name + ' is a FAILED RUN and will not be plotted')
        else:
            if ch.fileinfo.foil_name[0:9] in ['utest_unc']:
                key = 'Robust_'+str(i)
            else:
                key = 'Determinate_'+str(i)
            sel[key] = ch

    for key in sel.keys():
        kk = int(key[-1])
        data= np.loadtxt('../airfoils/' + sel[key].fileinfo.foil_name + '.txt', skiprows=1)
        sel[key].fileinfo.setcoords(data[:, 0], data[:, 1])
    # plot this shit
    mycols = ['firebrick','darksalmon','sienna','goldenrod','olive','limegreen','seagreen','teal','darkturquoise','dodgerblue','slategrey','blueviolet','purple','deeppink']
    import random
    cc = random.sample(range(0,len(mycols)), len(list(sel.keys())))
    cc = [0,2,4,6,-2]
    linestyle = ['-','--','-.',':','-','--','-.',':','-','--','-.',':','-','--','-.',':']
    ll = linestyle#[0:len(choice)]
    # fig1, (ax1,ax2,ax3) = plt.subplots(3)
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(9,7.5))
    gs = gridspec.GridSpec(2,2)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])
    ax3 = plt.subplot(gs[1,0])
    ax4 = plt.subplot(gs[1,1])
    # ax1.title.set_text(savename)
    ax1.title.set_text('Robust')
    ax2.title.set_text('Robust')
    ax3.title.set_text('Determinate')
    ax4.title.set_text('Determinate')
    ax1.grid(color = 'gainsboro')
    ax2.grid(color = 'gainsboro')
    ax3.grid(color='gainsboro')
    ax4.grid(color='gainsboro')
    # fig2, ax2 = plt.subplots(1)
    all_keys = list(sel.keys())
    for i in range(0,int(len(all_keys)/2)):
        if sel[all_keys[0]].fileinfo.reynolds_run:
            label = 'Mach: {:.2f}, Re: {:.1e}'.format(sel[all_keys[i]].input.mach, sel[all_keys[i]].input.reynolds)
        else:
            if sel[all_keys[i]].fileinfo.foil_name[0:9] in ['utest_unc']:
                label = 'Robust'
            else:
                label = 'Determinate'
        ax1.plot(sel[all_keys[i]].results.cd,sel[all_keys[i]].results.cl,mycols[cc[i]],linestyle=ll[i],label = label )
        if plot_type == 1:
            cl_cd = [sel[all_keys[i]].results.cl[j]/sel[all_keys[i]].results.cd[j] for j in range(0,len(sel[all_keys[i]].results.cl))]
            # ax3.plot(cl_cd, sel[k].results.cl, mycols[cc[i]],linestyle=ll[i])
            ax2.plot(sel[all_keys[i]].results.cl, cl_cd, mycols[cc[i]],linestyle=ll[i])
        elif plot_type == 2:
            ax2.plot(sel[all_keys[i]].results.alpha, sel[all_keys[i]].results.cl, mycols[cc[i]],linestyle=ll[i])

        ax3.plot(sel[all_keys[i+5]].results.cd, sel[all_keys[i+5]].results.cl, mycols[cc[i]],
                 linestyle=ll[i])  # ,label = sel[k].fileinfo.foil_name)]
        if plot_type == 1:
            cl_cd = [sel[all_keys[i+5]].results.cl[j] / sel[all_keys[i+5]].results.cd[j] for j in
                     range(0, len(sel[all_keys[i+5]].results.cl))]
            # ax3.plot(cl_cd, sel[k].results.cl, mycols[cc[i]],linestyle=ll[i])
            ax4.plot(sel[all_keys[i+5]].results.cl, cl_cd, mycols[cc[i]], linestyle=ll[i])
        elif plot_type == 2:
            ax4.plot(sel[all_keys[i+5]].results.alpha, sel[all_keys[i+5]].results.cl, mycols[cc[i]], linestyle=ll[i])
    # ax1.axis('equal')
    # ax2.legend(loc = 'lower right')
    ax1.set_xlabel(r'$C_d$',fontsize=14)
    ax1.set_ylabel(r'$C_l$',fontsize=14)
    ax1.set_ylim([0.0, 1.5])
    ax1.set_xlim([0.005, 0.045])
    ax3.set_xlabel(r'$C_d$', fontsize=14)
    ax3.set_ylabel(r'$C_l$', fontsize=14)
    ax3.set_ylim([0.0, 1.5])
    ax3.set_xlim([0.005, 0.045])
    if plot_type == 1:
        ax2.set_ylabel(r'$L/D$',fontsize=14)
        ax2.yaxis.set_ticks_position('right')
        ax2.yaxis.set_label_position('right')
        ax2.set_ylim([10,120])
        ax2.set_xlim([0.0,1.5])
        ax2.set_xlabel(r'$C_l$', fontsize=14)
        ax4.set_ylabel(r'$L/D$', fontsize=14)
        ax4.yaxis.set_ticks_position('right')
        ax4.yaxis.set_label_position('right')
        ax4.set_ylim([10, 120])
        ax4.set_xlim([0.0, 1.5])
        ax4.set_xlabel(r'$C_l$', fontsize=14)
    elif plot_type == 2:
        ax3.set_xlabel(r'$\alpha$',fontsize=14)
    # ax3.set_ylabel(r'$C_l$')
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0+box.height*0.25, box.width, box.height*0.9])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.9])
    # ax1.legend(loc='center left',fontsize=8, bbox_to_anchor=(0,-1),ncol=int(len(list(sel.keys()))/2))
    ncols = 3#len(list(sel.keys()))#3#min(4,int(len(list(sel.keys()))/2))
    ax1.legend(loc='upper center',bbox_to_anchor=(1,-0.175),fontsize=10,ncol=ncols)
    # plt.subplots_adjust(right=0.7)
    # plt.tight_layout()
    plt.savefig('../' + savename + '.png',dpi=250)
    plt.show()



''' ================================================================ '''
''' ================================================================ '''
''' ================================================================ '''
if __name__ == '__main__':

    '''
    RUN REQUIREMENTS
    1. airfoil files must be in ./airfoils/ folder
    2. script is called from ./lib/
    3. result data and images are saved in ./ root module directory
    4. if not using date-code method, change line 214 to read in whatever
        airfoil name you are using
    5. After data has been run, it is saved, and can be recalled by setting
        data_from_files = True, on line 202
    '''

    code    = 1213 # data code of data
    count   = 5 # number of attempted optimisation runs
    difficult = False # if struggling to converge sections
    ## USE ID to indicate which airfoils you want plotted
    id = 'utest_unc'
    id  = [id+str(code)+'_'+str(i) for i in range(0,count)]
    # id = ['utest_'+str(code)+'_2']
    id      = ['utest_1213_2', 'utest_unc_1213_4']#, 'utest_unc_1230_0', 'utest_unc_1230_2']
    config  =['utest_unc','utest']#, 'utest_unc', 'utest_unc2']

    re_airfoil = ['utest_unc_1213_4','utest_1213_2']
    # savename = id + '_comparison_' + str(code)
    # savename = savename + ' Re_{:.2e} M_{:.2f}'.format(reynolds, mach)

    # Get indiv not set up yet
    get_all = False # runs all airfoils in code, else just runs foils in id
    data_from_file=False # runs and generates plots, False calls from file data
    condition_change=False # run data to get changes for foils
    # run conds
    reynolds    = 1E6
    mach        = 0.2
    alpha       = [-3, 7, 0.1]
    cl          = [0.1, 2.0, 0.05]
    cl_ref      = 0.6



    if get_all:

        runs    = []
        entries = len(config)*count
        for i in range(0,entries):
            runs.append(Nexus())
        ct = 0
        for con in config:
            for i in range (0,count):
                savename = con +'_'+ str(code) +'_'+str(i)
                savename = savename + '-Re_{:.2e}-M_{:.2f}'.format(reynolds, mach)
                thisname = con+'_'+str(code)+'_'+str(i)
                runs[ct].fileinfo.set(thisname, code, i)
                runs[ct].fileinfo.difficult = difficult
                runs[ct].fileinfo.savename = savename
                runs[ct].input.set(mach, reynolds, alpha, cl, cl_ref, i)
                ct+=1
        # get results - read from files or call xfoil
        for i in range(0,len(runs)):
            input = runs[i].input
            fileinfo= runs[i].fileinfo
            results= runs[i].results
            if data_from_file:
                runs[i].results = read_results_to_nexus(runs[i])
            else:
                write_input(fileinfo, input)
                run_xfoil(fileinfo, input)
                runs[i].results = read_xfoil(fileinfo, input)
                write_results_to_file(fileinfo, runs[i].results)
        # plot results
        # for i in range(0,entries):
        plot_robust_comparison(runs, id, savename)
    elif not get_all and not condition_change:
        runs = []
        entries = len(id)
        for i in range(0, entries):
            runs.append(Nexus())
        ct = 0
        for con in id:
            i = int(con[-1])
            savename = con + '_' + str(code) + '_' + str(i)
            savename = savename + '-Re_{:.2e}-M_{:.2f}'.format(reynolds, mach)
            thisname = con
            runs[ct].fileinfo.set(thisname, code, i)
            runs[ct].fileinfo.difficult = difficult
            runs[ct].input.set(mach, reynolds, alpha, cl, cl_ref, i)
            ct += 1
        # get results - read from files or call xfoil
        for i in range(0,len(runs)):
            input = runs[i].input
            fileinfo= runs[i].fileinfo
            results= runs[i].results
            if data_from_file:
                runs[i].results = read_results_to_nexus(runs[i])
            else:
                write_input(fileinfo, input)
                run_xfoil(fileinfo, input)
                runs[i].results = read_xfoil(fileinfo, input)
                write_results_to_file(fileinfo, runs[i].results)
        # plot results
        # for i in range(0,entries):
        plot_robust_comparison(runs, id, savename)
    elif not get_all and condition_change:
        re_list = [ reynolds*(1 + (-0.3+i*1.5/10)) for i in range(0,5)]
        ma_list = [ mach*(1 + (-0.3+i*1.5/10)) for i in range(0,5)]
        # cl_list = [ [0.1, 1.3, 0.01], [0.1, 1.4, 0.01], [0.2, 1.4, 0.01], [0.3, 1.5, 0.01], [0.35, 1.6, 0.01]]
        cl_list = [[0.2, 1.3, 0.01], [0.3, 1.4, 0.01], [0.35, 1.4, 0.01], [0.4, 1.5, 0.01], [0.45, 1.6, 0.01]]
        id = []
        png = []
        for i in range(0,len(re_list)):
            # data storage set up
            runs = []
            entries = len(config) * count
            for i in range(0, entries):
                runs.append(Nexus())
            ct = 0
            # attribute run data
            for refoil in re_airfoil:
                for i in range(0, len(re_list)):
                    savename = refoil + '-Re_{:.0f}-M_{:.2f}'.format(re_list[i], ma_list[i])
                    id.append(savename)
                    thisname = refoil
                    runs[ct].fileinfo.set(thisname, code, i)
                    runs[ct].fileinfo.difficult = difficult
                    runs[ct].fileinfo.savename = savename
                    runs[ct].fileinfo.reynolds_run = True
                    runs[ct].input.set(ma_list[i], re_list[i], alpha, cl_list[i], cl_ref, int(refoil[-1]))
                    ct += 1
            #run each airfoil for range of reynolds numbers
        for i in range(0, len(runs)):
            input = runs[i].input
            fileinfo = runs[i].fileinfo
            results = runs[i].results
            if data_from_file:
                runs[i].results = read_results_to_nexus(runs[i])
            else:
                write_input(fileinfo, input)
                run_xfoil(fileinfo, input)
                runs[i].results = read_xfoil(fileinfo, input)
                write_results_to_file(fileinfo, runs[i].results)

        plot_robust_comparison2(runs, thisname+'_condvar_analysis')