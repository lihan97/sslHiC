import sys
sys.path.append('../')
import argparse
import numpy as np
from numpy.random import binomial
from scipy import sparse as sps
import pandas as pd
import cooler
from copy import deepcopy
from src.data.utils import cool_to_mats
str_to_num = {
    '1kb':1000, '5kb':5000, '10kb':10000, '25kb':25000,'40kb':40000,
    '50kb':50000, '100kb':100000, '250kb':250000, '500kb':500000, '1mb':1000000
}

def mats_to_cool(mats, noise_levels, noise_type, chrom, resol, out_dir):

    path = f"{out_dir}/{chrom}_{noise_type}.cool"
    print(f"Noisy contact matrices are saving to {path}")
    resol = str_to_num[resol]
    chrom_names = []
    chrom_sizes = []
    RAWobserved_list = []
    for mat, noise in zip(mats, noise_levels):
        assert len(mat) == len(mat[:,1])
        mat = sps.csc_matrix(mat)
        u, v = mat.toarray().nonzero()
        remain = (u<=v)
        bin1_id = u[remain]
        bin2_id = v[remain]
        count = mat.data[remain]
        RAWobserved = {
            'bin1_id':bin1_id,'bin2_id':bin2_id,'count':count
        }
        
        RAWobserved = pd.DataFrame(
            RAWobserved)
        RAWobserved['bin1_id'] = RAWobserved['bin1_id'].values + np.sum(chrom_sizes)
        RAWobserved['bin2_id'] = RAWobserved['bin2_id'].values + np.sum(chrom_sizes)
        RAWobserved_list.append(RAWobserved)
        chrom_names.append(f'{chrom}_{noise:.2f}')
        print(f"key in cooler file: {chrom}_{noise:.2f}")
        chrom_sizes.append(mat.shape[0])
        print(f"noise:{noise}, total weights:{np.sum(mat)}, num nodes:{mat.shape[0]}, num edges:{len( RAWobserved['count'])}, {len(RAWobserved['count'])}")
    pixels = pd.concat(RAWobserved_list)
    bins = pd.DataFrame({
        "chrom":np.concatenate([[chrom]*size for chrom, size in zip(chrom_names, chrom_sizes)]),
        "start": np.concatenate([np.arange(size)*resol for size in chrom_sizes]),
        "end": np.concatenate([np.arange(1,size+1)*resol for size in chrom_sizes])
    })
    cooler.create_cooler(path, bins, pixels, columns=['count'], dtypes={'count':np.float32})

def stratifiedSample ( V, F, strataSize = 100 ):
    N = len(V)
    V = np.array(V)
    F = np.array(F)

    strataCount = int(np.ceil(float(N) / strataSize))
    sortInd = np.argsort(F)
    strata = []
    strataMax = []



    for i in range(strataCount) :
        stratum = V [ sortInd[ (strataSize*(i) ) : (strataSize*(i+1)) ] ]
        stratumF = F [ sortInd[ (strataSize*(i) ) : (strataSize*(i+1)) ] ]
        strata.append( stratum )
        strataMax.append(max(stratumF))


    sample = []
    for i in range(len(V) ):
        if ( F[i] == 0 ) :
            sample.append (0)
        else :
            stratumInd = 0
            for k in range(strataCount) :
                if ( F[i] <= strataMax[k] ):
                    stratumInd = k
                    break
            if ( stratumInd == 0 ):
                stratumInd = k
            sample.append ( np.random.choice(strata[k],size=1)[0] )

    return ( sample )

def uniformMatrix ( CM, subSampleCount = 1000000, bias = True ):
    (R,C) = np.shape(CM)
    marginal = np.sum(np.array(CM),1)
    uniSampleCM = np.matrix( np.zeros((R,C)) )

    indexMap = []
    indexProb = []
    for i in range(R) :
        for k in range(i,R) :
            if marginal[i] != 0 and marginal[k] != 0 :
                indexMap.append([i,k])
                if bias :
                    indexProb.append(marginal[i] * marginal[k])
    if bias :
        totalProb = float(sum(indexProb))
        indexProb = [ iP / totalProb for iP in indexProb ]
        triuSample = np.random.choice(len(indexMap),subSampleCount,p=indexProb)
    else :
        triuSample = np.random.choice(len(indexMap),subSampleCount)

    for s in triuSample :
            (i,k) = indexMap[s]
            uniSampleCM[i,k] += 1
    uniSampleCM += np.transpose(np.triu(uniSampleCM,1))

    return (uniSampleCM)

def shuffleMatrix ( CM, stratumSize = 100 ):
    #Convert to integer
    CM = CM.astype(int)
    #Get marginals and number of rows
    contactSum = np.sum(np.array(CM),1)
    N = len(CM)

    # For matrix entry Mik, store Marginal i * Marginal k in CountByDist
    # and the Mik itself in matrixByDist
    countByDist = []
    matrixByDist = []
    for i in range(0,N):
        for k in range(i,N):
            dist = k-i
            if ( len(countByDist)-1 < dist ):
                countByDist.append( [ float(contactSum[i]) * contactSum[k] ] )
                matrixByDist.append( [ int( CM[i,k] ) ] )
            else:
                countByDist[dist].append( float(contactSum[i]) * contactSum[k] )
                matrixByDist[dist].append( int( CM[i,k] ) )

    noiseMatrix = np.zeros((N,N))

    for i in range(len(matrixByDist)):
    #for i in range(1):
        #print "dist is %d" % (i)
        thisSample = stratifiedSample(matrixByDist[i],countByDist[i],stratumSize)
        for k in range(len(thisSample)):
            noiseMatrix[k,k+i] = thisSample[k]
    for i in range(0,N):
        for k in range(i,N):
            noiseMatrix[k,i] = noiseMatrix[i,k]
    return ( noiseMatrix )

def simulate(ref_mat, noise_type, noise_levels):
    if noise_type == 'DropEdge':
        simulate_dropedge(ref_mat, noise_levels)
    elif noise_type == 'DropNode':
        noisy_mats = simulate_dropnode(ref_mat, noise_levels)
    elif noise_type == '66GD+33RL':
        noisy_mats = simulate_66GD(ref_mat, noise_levels)
    elif noise_type == '33GD+66RL':
        noisy_mats = simulate_33GD(ref_mat, noise_levels)
    else:
        NotImplementedError("Please specify the correct noise type")
    return noisy_mats
def simulate_dropedge(ref_mat, noise_levels):
    mats_de = []
    for noise in noise_levels:
        mat_de = np.triu(ref_mat)
        mat_de = sps.coo_matrix(mat_de)
        mat_de.data = mat_de.data * binomial(1,1-noise, size=len(mat_de.data))
        mat_de = mat_de.toarray()
        mat_de = np.triu(mat_de,0) + np.triu(mat_de,1).T
        mats_de.append(mat_de)
    return mats_de
def simulate_dropnode(ref_mat, noise_levels):
    nz_ids = np.where(np.sum(ref_mat, axis=1) != 0)[0]
    mats_dn = []
    for noise in noise_levels:
        mask = nz_ids[binomial(1,1-noise,size=len(nz_ids))==0]
        mat_dn = np.array(deepcopy(ref_mat))
        mat_dn[:,mask] = 0
        mat_dn[mask,:] = 0
        mats_dn.append(mat_dn)
    return mats_dn
def simulate_66GD(ref_mat, noise_levels):
    gdenm =  shuffleMatrix(np.triu(ref_mat))
    rlnm = uniformMatrix(ref_mat, subSampleCount=np.sum(np.triu(ref_mat)))
    mats_n = []
    for noise in noise_levels:
        mat_n = (1-noise) * ref_mat + noise * (2/3*gdenm + 1/3*rlnm)
        mats_n.append(mat_n)
    return mats_n
def simulate_33GD(ref_mat, noise_levels):
    gdenm =  shuffleMatrix(np.triu(ref_mat))
    rlnm = uniformMatrix(ref_mat, subSampleCount=np.sum(np.triu(ref_mat)))
    mats_n = []
    for noise in noise_levels:
        mat_n = (1-noise) * ref_mat + noise * (1/3*gdenm + 2/3*rlnm)
        mats_n.append(mat_n)
    return mats_n

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path',type=str, required=True, help='Path to reference Hi-C .cooler file')
    parser.add_argument("--resol", type=str, required=True, help= 'Resolution of Hi-C data')
    parser.add_argument("--chr", type=str, required=True, help='Target chromosome')
    parser.add_argument('--noise_type', type=str, required=True, help='Noise type to simulate', choices=['DropEdge', 'DropNode', '66GD+33RL', '33GD+66RL'])
    parser.add_argument('--noise_levels', type=str, required=False, default=[0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5])
    parser.add_argument('--out_dir', type=str, required=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ref_mat = cool_to_mats(args.ref_path, chroms=[args.chr])[0].toarray()
    noisy_mats = simulate(ref_mat, args.noise_type, args.noise_levels)
    mats_to_cool(noisy_mats, args.noise_levels, args.noise_type, args.chr, args.resol, args.out_dir)
