import numpy as np

def get_absolute_mags(logsysmass,age,met,binq,iso_int,bands):
    
    # Check whether this model is a binary and interpolate the isocrones to get the
    # intrinsic magnitudes of the current model

    sysmass = np.exp(logsysmass)
    nb = len(bands)

    if (binq >= 0):
        # If the system is binary get the mass ratio, obtain interpolated
        # magnitudes for each component and sum them
        mass_2  = sysmass/(1.+binq)*np.array([1.,binq],dtype=float)
        sysmags = np.nan*np.ones([2,nb])
        for j,mcomp in enumerate(mass_2):
            row = iso_int(mcomp,age,met)
            for i in range(nb):
                sysmags[j,i] = float(row[bands[i]])

        sysmags = 10**(-0.4*sysmags)
        absmags = np.nan*np.ones(nb)
        for j in range(nb):
            oneband = sysmags[:,j]
            msk = ~np.isnan(oneband)
            if (~np.any(msk)):
                absmags[j] = np.nan
            else:
                absmags[j] = -2.5* np.log10(np.sum(oneband[msk]))
    else:
        absmags = np.empty(nb)
        row = iso_int(sysmass,age,met)
        for i in range(nb):
            absmags[i] = float(row[bands[i]])

    return absmags

def get_observable_mags(absmags,DM=0.,ext=[0.,0.]):

    return absmags+ext+DM

def get_noisy_mags(obsmags,ASKDtree,AS_mag1_in,AS_mag2_in,AS_mag1_out,AS_mag2_out,AS_det):

    ind = ASKDtree.query(obsmags.reshape(1,-1),return_distance = False)[0]

    if (AS_det[ind]==True):
        m1 = AS_mag1_out[ind] -  AS_mag1_in[ind] + obsmags[0]
        m2 = AS_mag2_out[ind] -  AS_mag2_in[ind] + obsmags[1]
        
        return np.array([m1,m2]).flatten()       
    else:
        return np.nan*np.ones(2)
