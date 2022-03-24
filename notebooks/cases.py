import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import utilvolc.ash_inverse as ai
from utilhysplit.evaluation import ensemble_tools
from utilhysplit.plotutils import colormaker
from utilhysplit.evaluation import plume_stat

def set_lim(tii):
    if tii==16:
        xlim=(155,175)
        ylim=(46,61)
    elif tii==6:
        xlim=(157.5,163.5)
        ylim=(53.5,58)
    elif tii==8 or tii==7:
        xlim=(157.5,164)
        ylim=(52,58.5)
    else:
        xlim=(157.5,163)
        ylim=(54,58)
    return xlim, ylim



class EvalObj:
    def __init__(self,tag,aeval,
                 remove_cols=True, 
                 remove_rows=False, 
                 remove_sources=False, 
                 remove_ncs=5):
        self.tag = tag
        self.aeval = copy.copy(aeval)
        self.remove_cols=remove_cols
        self.remove_rows=remove_rows
        self.remove_sources = remove_sources
        self.remove_ncs = remove_ncs
        self.graphics_type = '.png'
        self.slope = None
        self.intercept = None

    def set_bias_correction(self,slope,intercept,dfcdf=pd.DataFrame(),cii=None):
        self.slope = slope
        self.intercept = intercept
        self.aeval.set_bias_correction(slope,intercept,dfcdf=dfcdf)
        self.cii = cii

    def reset_defaults(self):
        self.graphics_type = '.png'

    def maketaglist(self, tiilist):
        rlist = []
        for tii in tiilist:
            runtag = ai.create_runtag(self.tag,tii,self.remove_cols,
                                      self.remove_rows, self.remove_sources,
                                      remove_ncs = self.remove_ncs)
            rlist.append(runtag) 
        return rlist

    def maketaglistA(self):
        tiilist = [[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11]]
        return self.maketaglist(tiilist)   

    def maketaglistB(self):
        tiilist = [[2,3],[2,3,4],[2,3,4,5],[2,3,4,5,6],[2,3,4,5,6,7],[2,3,4,5,6,7,8],[2,3,4,5,6,7,8,9],
           [2,3,4,5,6,7,8,9,10],[2,3,4,5,6,7,8,9,10,11]]
        return self.maketaglist(tiilist)   

    def maketaglistall(self):
        tiilist = self.aeval.cdump.ens.values
        return tiilist        

    def set_thresholds(self,coarsen=0,
                       threshold=0.1,
                       pixel_match=False,
                       enslist=None):
        self.threshold = threshold
        self.coarsen = coarsen
        self.pixel_match = pixel_match
        self.threshstr = '{:0.1f}'.format(threshold).replace('.','p')
        self.enslist = enslist
        self.ctag = str(coarsen)
        if pixel_match: self.pmtag = '_pm'
        else: self.pmtag=''
        dft, dft2 = self.accuracy()


    def visualize(self,tii,vloc=None,cmap='viridis',verbose=True,figname=None):
        if verbose: print(self.aeval.cdump.time.values[tii])
        timestr = str(pd.to_datetime(self.aeval.get_time(tii))).replace(' ','_')
        timestr = timestr.replace(':00:00','')
        timestr = timestr.replace('-','')
        include='all'
        try:
            obs, model = self.aeval.get_pair(tii,coarsen=self.coarsen,cii=self.cii)
        except:
            return -1
        for ens in  model.ens.values:
            if verbose: print('ENS', ens)
            xlim, ylim = set_lim(tii)
            fig = self.aeval.compare_forecast(model.sel(ens=ens),vloc=vloc,cmap=cmap,xlim=xlim,ylim=ylim,
                                              include=include)

            if figname:
               plt.savefig(gdir + 'pcolormesh_run{}_{}_{}.png'.format(figname,timestr,ens))
            plt.show()
 
        # mean value
        if verbose: print('mean')
        fig = self.aeval.compare_forecast(model.mean(dim='ens'),vloc=vloc,cmap=cmap,xlim=xlim,ylim=ylim,
                                          include=include)
        if figname:
           plt.savefig(gdir + 'pcolormesh_run{}_{}_{}.png'.format(figname,timestr,'mean'))
        plt.show() 
        # max value
        if verbose: print('max')
        fig = self.aeval.compare_forecast(model.max(dim='ens'),vloc=vloc,cmap=cmap,xlim=xlim,ylim=ylim,
                                          include=include)
        if figname:
           plt.savefig(gdir + 'pcolormesh_run{}_{}_{}.png'.format(figname,timestr,'max'))
        plt.show() 
         

    def visualize_prob(self,tii,thresh,vloc=None,cmap='viridis',verbose=True,figname=None):
        #if verbose: print(self.aeval.cdump.time.values[tii])
        timestr = str(pd.to_datetime(self.aeval.get_time(tii))).replace(' ','_')
        timestr = timestr.replace(':00:00','')
        timestr = timestr.replace('-','')
        include='all'
        try:
            obs, model = self.aeval.get_pair(tii,coarsen=self.coarsen,cii=self.cii)
        except:
            return -1

        prob = ensemble_tools.ATL(model,thresh=thresh,norm=True)

        xlim, ylim = set_lim(tii)

        plevels = np.arange(0,1.1,0.15)
        plevels = plevels[1:-2]
        plevels = np.append(plevels,[1])
        clevels = [0.02,0.2,2,5,10]
        plevels = np.append([0.02],plevels)
        include=False
        fig = self.aeval.compare_forecast(prob,vloc=vloc,xlim=xlim,ylim=ylim,cmap_prob='cool',cmap='PuRd',
                            include=include,vmin=0,vmax=1,plevels=plevels,prob=True,clevels=clevels)




    def accuracy(self):
           
        volcat=[]
        forecast=[]
        # get_pair will return the cdump multiplied by the concmult factor set earlier.
        # get these times.
        for tii in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
            try:
                obs, model = self.aeval.get_pair(tii,coarsen=self.coarsen,cii=self.cii)
            except:
                #print('breaking accuracy at {} :'.format(tii))
                continue
            if isinstance(self.enslist,list):
                model = model.sel(ens=self.enslist)
            forecast.append(model)
            volcat.append(obs)
        # dft is a pandas dataframe with FSS information
        # dft2 is pandas dataframe with MSE, MAE information.
        dft, dft2 = ensemble_tools.ens_time_fss(forecast,volcat,threshold=self.threshold,
                                      neighborhoods=[1,3,5,7,9,11,13,15,17,19,21],plot=False,
                                      pixel_match=self.pixel_match)

        self.fssdf = dft
        self.accdf = dft2
        return dft, dft2


    def plot_areaA(self,enslist,tlist=[0.2,2],legend=True):
        sns.set()
        sns.set_style('whitegrid')
        fig = plt.figure(figsize=[20,5])
        nnn=1
        if legend: ccc=4
        else: ccc = 2
        ax1 = fig.add_subplot(nnn,ccc,1)
        ax2 = fig.add_subplot(nnn,ccc,2)
        #ax3 = fig.add_subplot(nnn,ccc,3)
        if legend: axg = fig.add_subplot(nnn,ccc,4)
        axlist = [ax1, ax2]
        for iii, thresh in enumerate(tlist):
            print(iii, len(tlist), thresh)
            self.set_thresholds(coarsen=0,threshold=thresh,
                                pixel_match=False, enslist=enslist)
            if iii < len(tlist)-1 or not legend:  
                self.plot_area(legend=False,ax=axlist[iii])
            else:
                self.plot_area(legend=True,ax=axlist[iii],axg=axg)
        return axlist   

    def plot_area(self,legend=False,ax=None,axg=None,enslist=None):
        clen = len(self.accdf.ens.unique())
        clrs = colormaker.ColorMaker('viridis',clen-1,ctype='hex',transparency=None)
        colors = clrs()
        colors = ['#'+x for x in colors]
        colors.append('#F03811')
        if not enslist: enslist = self.enslist.copy()
        #enslist.append('mean')
        ax = ensemble_tools.plot_ens_area(self.accdf,ax=ax,
                                          plotmean=False,legend=False,
                                          enslist = enslist,
                                          clrlist=colors) 
        handles, labels = ax.get_legend_handles_labels()
        if legend:
            if not axg:
               figlegend = plt.figure()
               axg = figlegend.add_subplot(1,1,1)
            axg.legend(handles,labels,loc="center",fontsize=20)
            axg.axis("off")
            #plt.savefig('emissions{}_legend.png'.format(tag))

    def makecolors(self,enslist=None):
        if not enslist:
           #clen = len(self.accdf.ens.unique())
           clen = len(self.enslist)
           print('num colors', clen)
        clrs = colormaker.ColorMaker('viridis',clen-1,ctype='hex',transparency=None)
        colors = clrs()
        colors = ['#'+x for x in colors]
        print(colors)
        return colors

    def plot_fssC(self,tii,df2,df3):
        sns.set_context('poster')
        fig = plt.figure(1,figsize=[20,5])
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)
        #ax4 = fig.add_subplot(1,4,4)

        # bias at 00z
        cii = datetime.datetime(2020,10,22,0)
        self.set_bias_correction(slope=None,intercept=None,dfcdf=df2,cii=cii)
        self.plot_fssB(tii,ax=ax2)

        # bias at current time
        # fourth plot
        #cii = datetime.datetime(2020,10,22,0)
        #self.set_bias_correction(slope=None,intercept=None,dfcdf=df2,cii=None)
        #self.plot_fssB(tii,ax=ax4)

        # bias at 00z with no positive intercepts
        # third plot
        cii = datetime.datetime(2020,10,22,0)
        self.set_bias_correction(slope=None,intercept=None,dfcdf=df3,cii=None)
        self.plot_fssB(tii,ax=ax3)

        # no bias
        # first plot
        self.set_bias_correction(slope=None,intercept=None,dfcdf=pd.DataFrame(),cii=None)
        self.plot_fssB(tii,ax=ax1)
        for ax in [ax1,ax2,ax3]:
            ax.set_ylim(0,1.0)
            ax.set_xlabel('degrees')
            ax.set_ylabel('FSS')

    def plot_fssB(self,tii=14, slope=None, intercept=None,ax=None):
        # plot fss as a function of neighborhood for given time, tii
        pixel_match=False
        threshold=0.2
        threshstr = str(threshold).replace('.','p')
        timeval = self.aeval.cdump.time.values[tii]
        print('time', timeval, 'threshold', threshold)
        volcat,forecast = self.aeval.get_pair(tii,slope=slope, intercept=intercept,cii=self.cii)
        #print(self.aeval.concmult)
        #forecast = aeval.cdump_hash[tii]
        nb = np.arange(1,21,2)
        nb = np.append(nb,[31,41,51])
        if tii > 10:
            nb = np.append(nb,[61,71,81,91,101])
        #nb = [11,81,91]
        # msc and psc are CalcScores objects.
        # msc is for the ensemble mean (deterinistic)
        # psc is for the probabilistic model field.
        msc, psc, df1, dfmae = ensemble_tools.ens_fss(forecast,volcat,threshold=threshold,
                                               neighborhoods=nb,
                                               return_objects=True,plot=False,
                                               pixel_match=pixel_match)
        ensemble_tools.plot_ens_fss(df1,xmult=0.1,ax=ax)
        ax = plt.gca()
        ax.set_ylim(0,1.0)
        ax.set_xlabel('degrees')
        ax.set_ylabel('FSS')
        if pixel_match: pmtag='_pm'
        else: pmtag = ''
        plt.tight_layout()
        return msc, psc, df1, dfmae




    def plot_afss(self):
        afss = ensemble_tools.plot_afss_ts(self.fssdf,clrs=self.makecolors())
        # no reason to save if pixel matching is True.
        #    plt.savefig(gdir + 'run{}_afss_ts_t{}{}.{}'.format(tag,threshold,biastag,graphicstype))
        #else:
        #    print('pixel match on')
 
    def plot_fssA(self,neighborhood=11):
        # plot fss as a function of time for given neighborhood
        sns.set()
        sns.set_style('whitegrid')
        # grid size is 0.1 degrees.
        # neighborhood gives the number of grid squares to
        # look at. 10 would be a 1degree x 1 degree area.
        clrs = self.makecolors()
        ensemble_tools.plot_ens_fss_ts(self.fssdf,nval=neighborhood,clrs=clrs)
        ax = plt.gca()
        ax.set_ylim(-0.01,0.98)
        plt.savefig('run{}_fss_ts_n{}_t{}{}.{}'.format(self.tag,
                                                       neighborhood,
                                                       self.threshstr,
                                                       self.pmtag,
                                                       self.graphics_type))


    def plot_accuracy(self):
        clen = len(self.accdf.ens.unique())
        clrs = colormaker.ColorMaker('viridis',clen-1,ctype='hex',transparency=None)
        colors = clrs()
        print(colors)
        colors = ['#'+x for x in colors]
        colors.append('#F03811')
        print(colors)


        enslist = self.enslist
        #enslist.append('mean')

        for val in ['MAE','MSE','RMSE','bias','fractional_bias','POD','FAR','CSI','B']:
            sns.set_style('whitegrid')
            ax = ensemble_tools.plot_ens_accuracy(self.accdf,val,
                                                  plotmean=False,
                                                  legend=False,
                                                  clrlist=colors,
                                                  enslist=enslist)
            if val in ['MAE','MSE','RMSE']:
                ax.set_yscale('log')
            newlab=[]
            handles, labels = ax.get_legend_handles_labels()
            #for lab in labels:
            #    newlab.append(handlehash[lab])
            newlab = getlabels()
            #if val in ['CSI']: ax.legend(handles, newlab)
            #ax.legend(handles, newlab)
            
            #d1 = datetime.datetime(2020,10,22,2)
            #x1 = [d1,d1]
            #y1 = [1e-1,1]
            #ax.plot(x1,y1,'--bo')
           
            plt.savefig('run{}_{}_t{}{}c{}.{}'.format(self.tag,val,
                                                      self.threshstr,self.pmtag,
                                                      self.ctag,self.graphics_type))
        plt.show()
        figlegend = plt.figure()
        axg = figlegend.add_subplot(1,1,1)
        axg.legend(handles,labels,loc="center",fontsize=20)
        axg.axis("off")
        #plt.savefig('emissions{}_legend.png'.format(tag))
        plt.show()
    

    def cdf_plot(self, d1, threshold=None):
        if not isinstance(d1,list): timelist=[d1]
        else: timelist = d1
        print(timelist)
        #print(enslist)l.
        if not threshold: threshold = self.threshold
        biastag=''
        if isinstance(d1,list):
            timestr = d1[0].strftime("%Y%m%dH%H")
        else:
            timestr = d1.strftime("%Y%m%dH%H")
        colors = self.makecolors()
        # CDF with pixel matching. Threshold will be different for every ensemble member as well as fo
        # volcat data. First threshold is applied to VOLCAT data. Number of VOLCAT pixels above threshold
        # is counted.
        figname = 'pixel_match_cdf_{}_{}{}.png'.format(self.tag,timestr,biastag)
        cdhash1 = self.aeval.mass_cdf_plot(timelist,
                            self.enslist,
                            threshold=threshold,
                            use_pixel_match=False,
                            plotdiff=False,
                            figname=figname,
                            colors=colors)
        #ax = plt.gca()
        #plt.savefig('pixel_match_ks_{}_{}{}.png'.format(self.tag,timestr,biastag))
        #plt.show()

        # CDF with no pixel matching. Thresholds are the same for every ensemble member and volcat data.
        # number of above threshold pixels will be different.
        #cdhash2 = self.aeval.mass_cdf_plot(timelist,
        #                    self.enslist,
        #                    threshold=threshold,
        #                    use_pixel_match=True,
        #                    plotdiff=False,
        #                    figname=figname,
        #                    colors=colors)
        #figname = 'cdf_{}_{}{}.png'.format(self.tag,timestr,biastag)
        #ax = plt.gca()
        #plt.savefig('ks_{}_{}{}.png'.format(self.tag,timestr,biastag))
        #plt.show()
        return cdhash1 


def describe_cdhash(cdhash):
    from scipy.stats import describe
    keys = list(cdhash.keys())
    date = []
    ens = []
    mean = []
    var = []
    skew = []
    kurt = []
    num = []
    small = []
    big = []
    
    for ky in keys:
        date.append(ky[0])
        ens.append(ky[1])
        sts = describe(cdhash[ky][0])
        mean.append(sts.mean)
        var.append(sts.variance)
        skew.append(sts.skewness)
        kurt.append(sts.kurtosis)
        num.append(sts.nobs)
        small.append(sts.minmax[0])
        big.append(sts.minmax[1])
        
    data = zip(date,ens,mean,var,skew,kurt,num,small,big)
    colnames = ['date','ens','mean','variance','skewness','kurtosis','N','min','max']
    dfout = pd.DataFrame(data)
    dfout.columns = colnames
    return dfout 
        

def getlabels(dt=[2,3,4,5,6,7,8,9,10]):
    d1 = datetime.datetime(2020,10,21,23)
    dlist = [d1+datetime.timedelta(hours=(n-2)*1) for n in dt]
    #print(dlist)
    labels = [x.strftime("%m/%d %H:00 UTC") for x in dlist]
    return labels

 
def getclrsGFS():
    clrs = []
    clrs.append(sns.xkcd_rgb['red']) 
    clrs.append(sns.xkcd_rgb['orange']) 
    clrs.append(sns.xkcd_rgb['orange']) 
    clrs.append(sns.xkcd_rgb['blue']) 
    clrs.append(sns.xkcd_rgb['green']) 

    clrs.append(sns.xkcd_rgb['orange']) 
    clrs.append(sns.xkcd_rgb['blue']) 
    clrs.append(sns.xkcd_rgb['green']) 
    clrs.append(sns.xkcd_rgb['cyan']) 

    clrs.append(sns.xkcd_rgb['orange']) 
    clrs.append(sns.xkcd_rgb['blue']) 
    clrs.append(sns.xkcd_rgb['dark teal']) 
    clrs.append(sns.xkcd_rgb['cyan']) 

    clrs.append(sns.xkcd_rgb['navy']) 

    return clrs

def set_lim(tii):
    if tii>=13:
        xlim=(155,170)
        ylim=(50,60)
    elif tii>=10:
        xlim=(157.5,167)
        ylim=(51.5,59)
    elif tii>=7:
        xlim=(157.5,164)
        ylim=(52,58.5)
    elif tii==6:
        xlim=(157.5,163.5)
        ylim=(53.5,58)
    else:
        xlim=(157.5,163)
        ylim=(54,58)
    return xlim, ylim 

def runFuego(tag='FA'):
    vhash={}
    vhash['vloc'] = [-90.88,14.473]
    vhash['tdir'] = '../Run{}'.format(tag)
    vhash['tname'] = 'xrfile.invFuegoA.nc'
    # location of volcat files
    vhash['vdir'] = '../data/volcatFuego/'.format(tag)
    # volcano id to locate
    vhash['vid'] = None
    vhash['gdir'] = './graphics/'
    vhash['graphicstype'] = 'png'
    return vhash

def runD():
    """
    Source from inversion algorithm using time periods 1,2,3,4.
    """
    vhash = {}
    vhash['vloc'] = [160.587,55.978]
    vhash['tag'] ='DI'
    tag = vhash['tag']

    #----------------------------------------------------------------
    # location and name of netcdf file with cdump output.
    vhash['tdir'] = '../Run{}'.format(tag)
    vhash['tname'] = 'runDI5.nc'

    #----------------------------------------------------------------
    vhash['configdir'] = '../Run{}'.format(tag)
    vhash['configfile'] = 'config.{}.txt'.format('di')

    #-----------------------------------------------------------------
    # location of volcat files
    vhash['vdir'] = '../data/volcat/'.format(tag)
    # volcano id to locate
    vhash['vid'] = 'v300250'
        
    #-----------------------------------------------------------------
    # Location to write graphics files
    vhash['gdir'] = './graphics/'
    vhash['graphicstype'] = 'png'
    return AshCase(vhash)

def runM(subset='M_2_3_4_TFw5'):
    """
    Source from inversion algorithm using time periods 1,2,3,4.
    """
    vhash = {}
    vhash['vloc'] = [160.587,55.978]
    vhash['tag'] ='M'
    tag = vhash['tag']

    #----------------------------------------------------------------
    # location and name of netcdf file with cdump output.
    vhash['tdir'] = '../Run{}/Run{}'.format(tag,subset)
    vhash['tname'] = 'RunM.nc'

    #----------------------------------------------------------------
    vhash['configdir'] = '../Run{}'.format(tag)
    vhash['configfile'] = 'config.invbezy{}.txt'.format(tag)

    #-----------------------------------------------------------------
    # location of volcat files
    vhash['vdir'] = '../data/volcat/'.format(tag)
    # volcano id to locate
    vhash['vid'] = 'v300250'
        
    #-----------------------------------------------------------------
    # Location to write graphics files
    vhash['gdir'] = './graphics/'
    vhash['graphicstype'] = 'png'
    return AshCase(vhash)

def runBcontrol():
    tag = 'BC'
    case = runA()
    case.vhash['tdir'] = '../RunB/control/'
    case.vhash['tname'] = 'RunBcontrol.nc'
    case.vhash['tag'] = tag
    case.vhash['configdir'] = '../Run{}'.format(tag)
    case.vhash['configfile'] = None
    return case

def runany(subset,tag):
    case = runA()
    if subset:
        case.vhash['tdir'] = '../Run{}/Run{}'.format(tag,subset)
    else:
        case.vhash['tdir'] = '../Run{}'.format(tag,subset)
    case.vhash['tname'] = 'Run{}.nc'.format(tag)
    case.vhash['tag'] = tag
    case.vhash['configdir'] = '../Run{}'.format(tag)
    case.vhash['configfile'] = 'config.invbezy{}.txt'.format(tag)
    return case

def runH():
    tag = 'H'
    case = runA()
    case.vhash['tdir'] = '../RunH/'
    case.vhash['tname'] = 'RunH_GFS.nc'
    case.vhash['tag'] = tag
    case.vhash['configdir'] = '../Run{}'.format(tag)
    case.vhash['configfile'] = 'config.invbezyH.txt'
    return case


def runB():
    tag = 'B'
    case = runA()
    case.vhash['tdir'] = '../RunB/'
    case.vhash['tname'] = 'RunB.nc'
    case.vhash['tag'] = tag
    case.vhash['configdir'] = '../Run{}'.format(tag)
    case.vhash['configfile'] = 'config.invbezyB.txt'
    return case

def runA2():
    """
    Cylindrical source
    """
    vhash = {}
    vhash['vloc'] = [160.587,55.978]
    vhash['tag'] ='A2'
    tag = vhash['tag']

    #----------------------------------------------------------------
    # location and name of netcdf file with cdump output.
    vhash['tdir'] = '../data/'
    vhash['tname'] = 'xrfile.ensCylBezyA2.nc'

    #----------------------------------------------------------------
    vhash['configdir'] = '../Run{}'.format(tag)
    vhash['configfile'] = 'config.ensCylBezyA2.txt'

    #-----------------------------------------------------------------
    # location of volcat files
    vhash['vdir'] = '../data/volcat/'.format(tag)
    # volcano id to locate
    vhash['vid'] = 'v300250'
        
    #-----------------------------------------------------------------
    # Location to write graphics files
    vhash['gdir'] = './graphics/'
    vhash['graphicstype'] = 'png'
    return AshCase(vhash)


def runA():
    """
    Cylindrical source
    """
    vhash = {}
    vhash['vloc'] = [160.587,55.978]
    vhash['tag'] ='A'
    tag = vhash['tag']

    #----------------------------------------------------------------
    # location and name of netcdf file with cdump output.
    vhash['tdir'] = '../data/'
    vhash['tname'] = 'xrfile.ensCylBezyA.nc'

    #----------------------------------------------------------------
    vhash['configdir'] = '../Run{}'.format(tag)
    vhash['configfile'] = 'config.ensCylBezyA.txt'

    #-----------------------------------------------------------------
    # location of volcat files
    vhash['vdir'] = '../data/volcat/'.format(tag)
    # volcano id to locate
    vhash['vid'] = 'v300250'
        
    #-----------------------------------------------------------------
    # Location to write graphics files
    vhash['gdir'] = './graphics/'
    vhash['graphicstype'] = 'png'
    return AshCase(vhash)


class AshCase:

    def __init__(self,vhash):
        self.vhash=vhash
 

    def __str__(self):
        rstr = ''
        for key in self.vhash.keys():
            rstr += '{} : {}\n'.format(key,self.vhash[key])      
        return rstr


def particle_mass(pmassmax):
    # need to compute pmassmax
    mult = cmult
    #cyldump = cyl.p060*mult
    print('mass of one particle {:2e} g'.format(pmassmax * mult))
    resolution = 0.1 # degrees
    lat = 56
    # area of one pixel.
    area = (resolution*111e3)**2 * np.cos(lat*np.pi/180.0)
    vres = 1.5 #km
    massload = pmassmax * mult / area
    conc = (massload*1000) / (vres*1000)
    print('Mass loading of one particle {:2e} g/m2'.format(massload))
    print('Concentration of one particle {:2e} mg/m3'.format(conc))
    print('Number of particles equal to 0.2 mg/m3 : {}'.format(0.2/conc))
    print('Number of particles equal to 2 mg/m3 : {}'.format(2/conc))


def plot_problistA(dtlist1, dtlist2, problist1, problist2,gdir='./',tag='',sz=(1,1)):
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)
        clr1 = ['--k','--m','--g','--c','--y','--r']
        clr2 = ['-k','-m','-g','-c','-y','-r']
        area1, base1 = plot_problist(dtlist1, problist1,ax,clr1,sz=sz[0])
        area2, base2 = plot_problist(dtlist2, problist2,ax,clr2,sz=sz[1])

        fig.savefig(gdir+'PRC_Run{}.png'.format(tag))

        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(1,1,1)
        ax2.plot(area1, '--ko')
        ax2.plot(base1, '--ro')
        ax2.plot(area2, '-ko')
        ax2.plot(base2, '-ro')
        return ax, ax2

        #plt.plot(arealist,'-k.')
        #plt.plot(blist,'--ro')
        #ax = plt.gca()
        #ax.set_ylim(0,1.05)

        


def plot_problist(dtlist, problist, 
                  ax=None,
                  clr = ['--ko','--mo','--go','--co','--yo','--ro'],
                  gdir='./', tag='',
                  sz=1):
    sns.set_style('whitegrid')
    sns.set_context('paper')
    if not ax:
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)
        dosave=True
    else:
        dosave=False
    arealist = []
    blist = []
    maxy = 0
    np.warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
    for iii, prob in enumerate(problist):
        label = dtlist[iii].strftime("%H UTC")
        plevels = np.arange(0.1,1.05,0.1)
        plevels = np.arange(0,32,1)
        plevels = plevels/31.0
        #plevels = np.append([0,1/31.0,2/31.0],plevels)
        print('highlight plevel',plevels[2],plevels[15])
        #print('All plevels', plevels)
        xlist, ylist,baseline,area = prob.calc_precision_recall(sz=sz,clip=False,problist=plevels)
        plume_stat.plot_precision_recall(xlist,ylist,float(baseline),ax=ax,clr=clr[iii],label=label)
        handles,labels = ax.get_legend_handles_labels()
        #ax.legend(handles,labels)
        ax.plot(xlist[2],ylist[2],'yo',MarkerSize=10,alpha=0.5)
        ax.plot(xlist[15],ylist[15],'ro',MarkerSize=10,alpha=0.5)
        #print(area, float(baseline))
        arealist.append(area)
        blist.append(baseline)
        maxy = np.max([maxy,np.max(ylist)])
        #print(maxy)
    ax.set_ylim(0,maxy+0.1)
    if dosave:
        plt.savefig(gdir+'PRC_Run{}.png'.format(tag))
        plt.show()

        plt.plot(arealist,'-k.')
        plt.plot(blist,'--ro')
        ax = plt.gca()
        #ax.set_ylim(0,1)
    return arealist, blist
 

def make_rank(aeval,thresh,cii=None,coarsen=None,coarsen_max=None,tlist=None):
    from utilhysplit.evaluation import reliability
    
    #thresh=0.2
    threshstr = str(thresh).replace('.','p')
    nbins=32
    rank1 = reliability.Talagrand(thresh,nbins)
    if not tlist: tlist = [4,5,6,7,8,9,10,11,12,13] 
    for tii in tlist:
        #volcat = aeval.volcat_avg_hash[tii]
        #forecast = aeval.cdump_hash[tii]
        volcat, forecast = aeval.get_pair(tii,coarsen=coarsen,cii=cii,coarsen_max=coarsen_max)
        dfin = rank1.add_data_xra(volcat,forecast)
    return rank1

def relrank_final_plots(aeval,df2,df3,tlist=[4,5,6]):
    sns.set_style('whitegrid')
    sns.set_context('paper')
    fig = plt.figure(figsize=[20,5])
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    #ax4 = fig.add_subplot(1,5,4)
    #ax5 = fig.add_subplot(1,5,5)


    coarsen=None
    coarsen_max=None

    cii=datetime.datetime(2020,10,22,0)
    rank_thresh=0.1

    aeval.set_bias_correction(slope=None, intercept=None, dfcdf=pd.DataFrame())
    rank = make_rank(aeval,rank_thresh,None,coarsen,coarsen_max,tlist)  
    rank.plotrank(ax=ax1)
    #plot_reliability(aeval,cii,ax4,ax5,coarsen_max,tag='',tlist=tlist)
    aeval.set_bias_correction(slope=None, intercept=None, dfcdf=df2)
    rank = make_rank(aeval,rank_thresh,cii,coarsen,coarsen_max,tlist)  
    rank.plotrank(ax=ax1)

    plot_reliability(aeval,cii,ax2,ax3,coarsen_max,tag='',tlist=tlist)
    return ax1,ax2,ax3,fig

def relrank_final_plotsB(aeval,df2,clrs = ['-m','-c','-y']):
    if colorset==1:
        clrs = ['--m','--c','--y']
    elif colorset==2:
        clrs = ['-m','-c','-y']
    rclist = []

    labels=[]
    for thresh in threshlist:
        if isinstance(thresh,(float,int)):
            labels.append(str(thresh))
        else:
            labels.append('{} to {}'.format(thresh[0],thresh[1]))
        rclist.append(reliability.ReliabilityCurve(thresh,num))

    for tii in tlist:
        volcat,forecast = aeval.get_pair(tii,cii=cii,coarsen_max=coarsen_max)
        for jjj, rc in enumerate(rclist):
            dfin = rc.reliability_add_xra(volcat,forecast,fill=True)
    for jjj, rc in enumerate(rclist):
        reliability.sub_reliability_plot(rc,ax,clr=clrs[jjj],fs=10,label=labels[jjj])
        reliability.sub_reliability_number_plot(rc,ax2,clr=clrs[jjj],fs=10,label=labels[jjj])


def plot_reliability(aeval, cii, ax=None, ax2=None, 
                     coarsen_max=None,tag='',gdir='./',tlist=[4,5,6,7,8,9,11,11]):
    from utilhysplit.evaluation import reliability
    num=15
    sns.set()
    sns.set_context('talk')
    threshlist = [0.1,0.2,[0.1,2],2,[2,5],5.0]
    threshlist = [0.1,0.2,2]
    clrs = ['--m','-c','-y','-g','-co','-y']
    rclist = []
    labels = []
    if not ax:
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(1,1,1)
    for thresh in threshlist:
        if isinstance(thresh,(float,int)):
            labels.append(str(thresh))
        else:
            labels.append('{} to {}'.format(thresh[0],thresh[1]))
        rclist.append(reliability.ReliabilityCurve(thresh,num))
    # time periods to include in reliability diagram.
    #tlist = [2,3,4,5,6,7]
    for tii in tlist:
    #for tii in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
        volcat,forecast = aeval.get_pair(tii,cii=cii,coarsen_max=coarsen_max)
        for jjj, rc in enumerate(rclist):
            dfin = rc.reliability_add_xra(volcat,forecast,fill=True)

    for jjj, rc in enumerate(rclist):
        reliability.sub_reliability_plot(rc,ax,clr=clrs[jjj],fs=10,label=labels[jjj])
        reliability.sub_reliability_number_plot(rc,ax2,clr=clrs[jjj],fs=10,label=labels[jjj])

    rel_time_str = str.join('_',map(str,tlist))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,labels,loc='upper left')

    #fig.savefig(gdir + 'reliability_{}_t{}'.format(tag,rel_time_str))
    #fig2.savefig(gdir + 'reliability_number_{}_t{}'.format(tag,rel_time_str))





