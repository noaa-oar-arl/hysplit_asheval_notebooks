
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

def runM():
    """
    Source from inversion algorithm using time periods 1,2,3,4.
    """
    vhash = {}
    vhash['vloc'] = [160.587,55.978]
    vhash['tag'] ='M'
    tag = vhash['tag']

    #----------------------------------------------------------------
    # location and name of netcdf file with cdump output.
    vhash['tdir'] = '../Run{}/Run{}'.format(tag,'M_2_3_4_TFw5')
    vhash['tname'] = 'RunM4.nc'

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
    vhash['tname'] = 'xrfile.ensCylBezy.nc'

    #----------------------------------------------------------------
    vhash['configdir'] = '../Run{}'.format(tag)
    vhash['configfile'] = 'config.ensCylBezy.txt'

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











