import sys
sys.path.append('../../PBVI_numpyBased/src/')
import numpy as np
from pomdpNumpy import Backup, GetBetaA, GetBetaAO, StateEstimator, argmaxAlpha, getPolicy


class Perseus(object):
    
    def __init__(self, perseusBackup, convergenceTolerance, V):
        self.perseusBackup=perseusBackup
        self.convergenceTolerance=convergenceTolerance
        self.V=V
        
    def __call__(self, B):
        delta=np.inf
        V=self.V
        vNew=np.dot(V['alpha'], B.T).max(axis=0)
        while (delta > self.convergenceTolerance):
            v=vNew.copy()
            VNew=self.perseusBackup(B, V)
            vNew=np.dot(VNew['alpha'], B.T).max(axis=0)
            print(max(vNew))
            delta=abs(vNew-v).max()
            print('d: ', delta)
            V=VNew.copy()
        return V


class PerseusBackup(object):
    
    def __init__(self, selectBelief, updateB, updateV):
        self.selectBelief=selectBelief
        self.updateV=updateV
        self.updateB=updateB
        
    def __call__(self, B, V):
        VNew={}
        BTilde=B.copy()
        while BTilde.size != 0:
            b=self.selectBelief(BTilde)
            VNew=self.updateV(b, VNew, V)
            BTildeNew=self.updateB(BTilde, VNew, V) 
            BTilde=BTildeNew.copy()
            print(BTilde.shape[0])
        return VNew
            
    
class UpdateV(object):
    
    def __init__(self, backup):
        self.backup=backup
        
    def __call__(self, b, VNew, V):
        alphaNew=self.backup(V, b)
        alphaNewTimesBValue=np.dot(b, alphaNew['alpha'])
        index=np.argmax(np.dot(V['alpha'], b))
        alpha={'action': V['action'][index], 'alpha': V['alpha'][index]}
        alphaTimesBValue=np.dot(b, alpha['alpha'])
        if alphaNewTimesBValue>=alphaTimesBValue:
            if VNew.get('alpha', np.array([])).size==0:
                VNew={'action': np.array([alphaNew['action']]), 'alpha': np.array([alphaNew['alpha']])}
            else:
                VNew['alpha']=np.vstack((VNew['alpha'], alphaNew['alpha']))
                VNew['action']=np.append(VNew['action'], alphaNew['action'])
        else:
            if VNew.get('alpha', np.array([])).size==0:
                VNew={'action': np.array([alpha['action']]), 'alpha': np.array([alpha['alpha']])}
            else:
                VNew['alpha']=np.vstack((VNew['alpha'], alpha['alpha']))
                VNew['action']=np.append(VNew['action'], alpha['action'])
        return VNew
   



def updateB(B, VNew, V):
    alphaNewTimesBValue=np.dot(VNew['alpha'], B.T).max(axis=0)
    alphaTimesBValue=np.dot(V['alpha'], B.T).max(axis=0)
    BNew=B[np.round(alphaNewTimesBValue, 8)<np.round(alphaTimesBValue, 8)]
    return BNew
   

def main():
    
    transitionMatrix=np.array([[[0.5, 0.5], [0.5, 0.5], [1, 0]],
                               [[0.5, 0.5], [0.5, 0.5], [0, 1]]])
    rewardMatrix=np.array([[[-100, -100], [10, 10],     [-1, -1]],
                           [[10, 10],     [-100, -100], [-1, -1]]])
    observationMatrix=np.array([[[0, 0, 1], [0, 0, 1], [0.85, 0.15, 0]],
                                [[0, 0, 1], [0, 0, 1], [0.85, 0.15, 0]]])
    
    beliefTransition=StateEstimator(transitionMatrix, observationMatrix)
    getBetaAO=GetBetaAO(beliefTransition, argmaxAlpha)
    
    gamma=0.95
    getBetaA=GetBetaA(getBetaAO, transitionMatrix, rewardMatrix, observationMatrix, gamma)
    backup=Backup(getBetaA, transitionMatrix)

    V={'action': np.array([2]), 'alpha': np.array([[rewardMatrix.min()/(1-gamma) for s in range(transitionMatrix.shape[0])]])}
    
    updateV=UpdateV(backup)    
    selectBelief=lambda B: B[np.random.choice(len(B))]
    perseusBackup=PerseusBackup(selectBelief, updateB, updateV)
    
    convergenceTolerance=1e-5
    perseus=Perseus(perseusBackup, convergenceTolerance, V)
    
    B=np.array([[0.05*n, 1-0.05*n] for n in range(21)])
    VNew=perseus(B)
    print(VNew)
    
    a=[getPolicy(VNew, b) for b in B]
    print(a)

if __name__=="__main__":
    main()        
