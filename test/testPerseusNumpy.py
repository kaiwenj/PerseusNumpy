
import sys
sys.path.append('../src/')
import numpy as np

import unittest
from ddt import ddt, data, unpack
from numpy.testing import assert_almost_equal
import perseusNumpy as targetCode

@ddt
class TestPerseusBackup(unittest.TestCase):
    
    @data((lambda B: B[0], lambda b, VNew, V: V, lambda B, VNew, V: np.array([b-1 for b in B if b>V[0]]), 
           np.array([1, 2, 3, 4, 5]), [2, 3], np.array([2, 3])))
    @unpack
    def testPerseusBackupOnlyChangeB(self, selectBelief, updateV, updateB, B, V, expectedResult):
        perseusBackup=targetCode.PerseusBackup(selectBelief, updateB, updateV)
        calculatedResult=perseusBackup(B, V)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data((lambda B: B[0], lambda b, VNew, V: [v+1 for v in V], lambda B, VNew, V: np.array([b-1 for b in B if b>V[0]]), 
           np.array([1, 2, 3, 4, 5]), [2, 3], np.array([3, 4])))
    @unpack
    def testPerseusBackupVChanges(self, selectBelief, updateV, updateB, B, V, expectedResult):
        perseusBackup=targetCode.PerseusBackup(selectBelief, updateB, updateV)
        calculatedResult=perseusBackup(B, V)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data((lambda B: B[0], lambda b, VNew, V: [v+1 for v in V], lambda B, VNew, V: np.array([b-1 for b in B if b>V[0]]), 
           np.array([1, 2, 3, 4, 5]), [1, 2], np.array([2, 3])))
    @unpack
    def testPerseusBackupVNewDoesNotChange(self, selectBelief, updateV, updateB, B, V, expectedResult):
        perseusBackup=targetCode.PerseusBackup(selectBelief, updateB, updateV)
        calculatedResult=perseusBackup(B, V)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data((lambda B: B[0], lambda b, VNew, V: [-2], lambda B, VNew, V: np.array([b-2 for b in B if b>VNew[0]]), 
           np.array([1, 2, 3, 4, 5]), [1, 2], np.array([-2])))
    @unpack
    def testPerseusBackupVNewBSynergize(self, selectBelief, updateV, updateB, B, V, expectedResult):
        perseusBackup=targetCode.PerseusBackup(selectBelief, updateB, updateV)
        calculatedResult=perseusBackup(B, V)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data((lambda B: min(B), lambda b, VNew, V: [b], lambda B, VNew, V: np.array([b-2 for b in B if b>VNew[0]]), 
           np.array([1, 2, 3, 4, 5]), [1, 2], np.array([-3])))
    @unpack
    def testPerseusBackupVNewBSelection(self, selectBelief, updateV, updateB, B, V, expectedResult):
        perseusBackup=targetCode.PerseusBackup(selectBelief, updateB, updateV)
        calculatedResult=perseusBackup(B, V)
        assert_almost_equal(calculatedResult, expectedResult)
        
    def tearDown(self):
        pass


@ddt
class TestUpdateV(unittest.TestCase):
    
    @data((lambda V, b: {'action': 1, 'alpha': np.array([10, 20])}, np.array([1, 3]), {}, 
           {'action': np.array([1, 2, 3, 4, 5]), 'alpha': np.array([[1, 2], [6, 7], [4, 7], [6, 7], [3, 2]])}, 
           {'action': np.array([1]), 'alpha': np.array([[10, 20]])}))
    @unpack
    def testUpdateVSuccess(self, backup, b, VNew, V, expectedResult):
        updateV=targetCode.UpdateV(backup)
        calculatedResult=updateV(b, VNew, V)
        assert_almost_equal(calculatedResult['action'], expectedResult['action'])
        assert_almost_equal(calculatedResult['alpha'], expectedResult['alpha'])
        
    @data((lambda V, b: {'action': 5, 'alpha': np.array([20, 20])}, np.array([1, 3]), 
           {'action': np.array([1]), 'alpha': np.array([10, 20])}, 
           {'action': np.array([1, 2, 3, 4, 5]), 'alpha': np.array([[1, 2], [6, 7], [4, 7], [6, 7], [3, 2]])}, 
           {'action': np.array([1, 5]), 'alpha': np.array([[10, 20], [20, 20]])}))
    @unpack
    def testUpdateAddOn(self, backup, b, VNew, V, expectedResult):
        updateV=targetCode.UpdateV(backup)
        calculatedResult=updateV(b, VNew, V)
        assert_almost_equal(calculatedResult['action'], expectedResult['action'])
        assert_almost_equal(calculatedResult['alpha'], expectedResult['alpha'])
    
    @data((lambda V, b: {'action': 5, 'alpha': np.array([2, 2])}, np.array([1, 3]), 
           {}, 
           {'action': np.array([1, 2, 3, 4, 5]), 'alpha': np.array([[1, 2], [6, 7], [4, 7], [6, 7], [3, 2]])}, 
           {'action': np.array([2]), 'alpha': np.array([[6, 7]])}))
    @unpack
    def testUpdateEmptyFail(self, backup, b, VNew, V, expectedResult):
        updateV=targetCode.UpdateV(backup)
        calculatedResult=updateV(b, VNew, V)
        assert_almost_equal(calculatedResult['action'], expectedResult['action'])
        assert_almost_equal(calculatedResult['alpha'], expectedResult['alpha'])
        
    @data((lambda V, b: {'action': 5, 'alpha': np.array([2, 2])}, np.array([1, 3]), 
           {'action': np.array([1]), 'alpha': np.array([10, 20])}, 
           {'action': np.array([1, 2, 3, 4, 5]), 'alpha': np.array([[1, 2], [6, 7], [4, 7], [6, 7], [3, 2]])}, 
           {'action': np.array([1, 2]), 'alpha': np.array([[10, 20], [6, 7]])}))
    @unpack
    def testUpdateAddOnFail(self, backup, b, VNew, V, expectedResult):
        updateV=targetCode.UpdateV(backup)
        calculatedResult=updateV(b, VNew, V)
        assert_almost_equal(calculatedResult['action'], expectedResult['action'])
        assert_almost_equal(calculatedResult['alpha'], expectedResult['alpha'])
 
    def tearDown(self):
        pass
    

@ddt
class TestUpdateB(unittest.TestCase):
    
    @data((np.array([[1, 2], [2, 3], [4, 5]]), {'action': np.array([1, 2, 3, 4, 5]), 'alpha': np.array([[1, 2], [6, 7], [4, 7], [6, 7], [3, 2]])},
           {'action': np.array([1, 2, 3, 4, 5]), 'alpha': np.array([[1, 2], [6, 7], [4, 7], [6, 7], [3, 2]])}, np.empty(shape=(0, 2))))
    @unpack
    def testUpdateBSame(self, B, VNew, V, expectedResult):
        calculatedResult=targetCode.updateB(B, VNew, V)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data((np.array([[10, 2], [2, 3], [4, 5]]), {'action': np.array([1, 2, 3, 4, 5]), 'alpha': np.array([[1, 2], [6, 7], [4, 7], [6, 7], [3, 2]])},
           {'action': np.array([1, 2, 3, 4, 5]), 'alpha': np.array([[10, 2], [6, 7], [4, 7], [6, 7], [3, 2]])}, np.array([[10, 2]])))
    @unpack
    def testUpdateBSome(self, B, VNew, V, expectedResult):
        calculatedResult=targetCode.updateB(B, VNew, V)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data((np.array([[2, 10], [2, 3], [4, 5]]), {'action': np.array([1, 2, 3, 4, 5]), 'alpha': np.array([[1, 2], [6, 7], [4, 7], [6, 7], [3, 2]])},
           {'action': np.array([1, 2, 3, 4, 5]), 'alpha': np.array([[10, 2], [6, 7], [4, 7], [6, 7], [3, 2]])}, np.empty(shape=(0, 2))))
    @unpack
    def testUpdateBNone(self, B, VNew, V, expectedResult):
        calculatedResult=targetCode.updateB(B, VNew, V)
        assert_almost_equal(calculatedResult, expectedResult)
        
    @data((np.array([[2, 10], [2, 3], [4, 5]]), {'action': np.array([1, 2, 3, 4, 5]), 
                                                 'alpha': np.array([[-1, -2], [-6, -7], [-4, -7], [-6, -7], [-3, -2]])},
           {'action': np.array([1, 2, 3, 4, 5]), 'alpha': np.array([[10, 2], [6, 7], [4, 7], [6, 7], [3, 2]])}, 
           np.array([[2, 10], [2, 3], [4, 5]])))
    @unpack
    def testUpdateBAll(self, B, VNew, V, expectedResult):
        calculatedResult=targetCode.updateB(B, VNew, V)
        assert_almost_equal(calculatedResult, expectedResult)
            
    def tearDown(self):
        pass


@ddt
class TestPerseus(unittest.TestCase):
    
    @data((lambda B, V: {'action': np.array([1]), 'alpha': V['alpha']/10}, 1, np.array([[1, 2, 3], [3, 5, 7]]), 
           {'action': np.array([1, 2, 3]), 'alpha': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])}, 
           {'action': np.array([1]), 'alpha': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])/10000}))
    @unpack
    def testPerseusChange(self, perseusBackup, convergenceTolerance, B, V, expectedResult):
        perseus=targetCode.Perseus(perseusBackup, convergenceTolerance, V)
        calculatedResult=perseus(B)
        assert_almost_equal(calculatedResult['action'], expectedResult['action'])
        assert_almost_equal(calculatedResult['alpha'], expectedResult['alpha'])
        
    @data((lambda B, V: {'action': np.array([1]), 'alpha': np.vstack((V['alpha']/10, np.array([0, 0, 0])))}, 1, np.array([[1, 2, 3], [3, 5, 7]]), 
           {'action': np.array([1, 2, 3]), 'alpha': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])}, 
           {'action': np.array([1]), 'alpha': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])/10000}))
    @unpack
    def testPerseusAdd(self, perseusBackup, convergenceTolerance, B, V, expectedResult):
        perseus=targetCode.Perseus(perseusBackup, convergenceTolerance, V)
        calculatedResult=perseus(B)
        assert_almost_equal(calculatedResult['action'], expectedResult['action'])
        assert_almost_equal(calculatedResult['alpha'], expectedResult['alpha'])
        
        
            
    def tearDown(self):
        pass




if __name__ == '__main__':
	unittest.main(verbosity=2)






