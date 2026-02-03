#!/usr/env/python python
import numpy as np
from pyscf import gto, scf, dft

def collect_Matrix(source_matrix, row, column):
    object_matrix = [[0.0 for i in range(len(column))] for j in range(len(row))]
    for i in range(len(row)):
        for j in range(len(column)): 
            object_matrix[i][j] = source_matrix[row[i]][column[j]]            
    return np.array(object_matrix)
'''
Compute oscilator strength and transition dipole moment from a delta-SCF excited state.
'''

'''
transition_momentum
INPUT:
mol:      PySCF molecular structure 
gs_mf:    Ground state SCF object from PySCF
ex_mf:    Delta SCF excited state SCF object from PySCF

OUTPUT:

'''
def transition_momentum(mol, gs_mf, ex_mf):
    # Determine which type of GTOs is utilized in calculation
    isCartesian = mol.cartesian
    
    # Obtain MO coefficients and electron occupations from ground state
    gs_coeff = gs_mf.mo_coeff
    gs_occ   = gs_mf.mo_occ

    # Obtain MO coefficients and electron occupations from delta SCF excited state
    ex_coeff = ex_mf.mo_coeff
    ex_occ   = ex_mf.mo_occ

    # Compute integrals in AO space
    if isCartesian:
        PMatrix_AO = mol.intor('int1e_ipovlp_cart')
        ovlp_matrix_AO = mol.intor('int1e_ovlp_cart')
        if mol.verbose > 5:
            print("Using cartesian GTOs")
        
    else :
        PMatrix_AO = mol.intor('int1e_ipovlp')
        ovlp_matrix_AO = mol.intor('int1e_ovlp')
        if mol.verbose > 5:
            print("Using spherical GTOs")

    # Initialize 
    determinant_S21 = [1.0, 1.0]
    dipole_determinant = [[], []]
    row_ground = [[], []]
    column_ex = [[], []]
    
    # Determine the index of excitations
    for i_spin in range(2):
        row_ground[i_spin] = []
        for i_state in range(len(gs_occ[i_spin])):
            if abs(gs_occ[i_spin][i_state] - 1.0) < 1.e-5 :
                row_ground[i_spin].append(i_state)
    
        column_ex[i_spin] = []
        for i_state in range(len(ex_occ[i_spin])):
            if abs(ex_occ[i_spin][i_state] - 1.0) < 1.e-5 :
                column_ex[i_spin].append(i_state)
        
        assert len(column_ex[i_spin]) == len(row_ground[i_spin]), \
            "Numbers of electrons in one spin channel must be equal to have a non-zero contribution!"
        
        dipole_determinant[i_spin] = [[1.0, 1.0, 1.0] for _ in range(len(row_ground[i_spin]))] 
        
        # row_ground must be equal to column_1st, 
        # otherwise the transition dipole moment 
        # must be zero.
        
        # S = <GS|EX>
        S21_temp = np.dot(gs_coeff[i_spin].T, ovlp_matrix_AO)
        S21 = np.dot(S21_temp, ex_coeff[i_spin])
        S21_reduced = collect_Matrix(S21, row_ground[i_spin], column_ex[i_spin])
        
        
        # P = <GS|p|EX>
        P21_temp = []
        P21_reduced = [[] for i_coord in range(3)]
        for i_coord in range(3):
            P21_temp = np.dot(gs_coeff[i_spin].T, PMatrix_AO[i_coord])
            P21 = np.dot(P21_temp, ex_coeff[i_spin])
            P21_reduced[i_coord] = collect_Matrix(P21, row_ground[i_spin], column_ex[i_spin])
        
        Um, Sigma, Vdagger = np.linalg.svd(S21_reduced, full_matrices=True, \
            compute_uv = True, hermitian = False)
        
        
        print('The singular values of overlap matrix for spin = ', i_spin)
        print(Sigma)
    # Note this overlap matrix is no way hermitian.
    # Make sure hermitian is specified correctly.
    # By the way, the default is non-hermitian.
    
        Um_dagger = Um.conjugate().T
        Vm = Vdagger.conjugate().T
        for i_coord in range(3): # x,y,z
            temp1 = np.dot(Um_dagger, P21_reduced[i_coord])
            temp2 = np.dot(temp1, Vm) 
            for i_state in range(len(row_ground[i_spin])):
                dipole_determinant[i_spin][i_state][i_coord] = temp2[i_state,i_state]
                for j_state in range(len(row_ground[i_spin])):
                    if i_state != j_state :
                        dipole_determinant[i_spin][i_state][i_coord] *= Sigma[j_state]
        # only Sigma and P21_reduced are needed for computing 
        # the transition dipole moment in the next step.
    
        for i_state in range(len(Sigma)):
            determinant_S21[i_spin] *= Sigma[i_state]
    
    final_sum = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    
    for i_spin in range(2):
        for i_state in range(len(row_ground[i_spin])):
            for i_coord in range(3):
               final_sum[i_spin][i_coord] += dipole_determinant[i_spin][i_state][i_coord]
    
    for i_coord in range(3):
        final_sum[1][i_coord] *= determinant_S21[0]
        final_sum[0][i_coord] *= determinant_S21[1]
            
    print(final_sum[0])
    return final_sum

if __name__ == "__main__":
    mol = gto.Mole()
    mol.verbose = 5
    #mol.output ='mom_DeltaSCF.out'
    mol.atom = [
        ["H", ( 0.000000, 0.000000, -0.52900)],
        ["H", ( 0.000000, 0.000000,  0.52900)]]
    mol.basis = {"H": 'aug-cc-pvtz'}

    mol.build()
    mol.symmetry = True
    mol.cartesian = False

    # 1. mom-Delta-SCF based on unrestricted HF/KS 
    a = dft.UKS(mol)
    a.xc = 'cam-b3lyp'
    # Use chkfile to store ground state information and start excited state
    # caculation from these information directly 
    a.scf()

    # Read MO coefficients and occpuation number from chkfile
    mo0 = a.mo_coeff
    occ = a.mo_occ
    
    # Determine HOMO
    nelec = np.floor(np.sum(occ[0])+1e-10)
    i_state = int(nelec) # Lowerest excited state, HOMO(alpha) -> LUMO(alpha)
   
    # Assign initial occupation pattern
    occ[0][i_state-1]=0      # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
    occ[0][i_state]=1      # it is still a singlet state

    # New SCF caculation 
    b = dft.UKS(mol)
    b.xc = 'cam-b3lyp'

    # Construct new dnesity matrix with new occpuation pattern
    dm_u = b.make_rdm1(mo0, occ)
    # Apply mom occupation principle
    b = scf.addons.mom_occ(b, mo0, occ)
    # Start new SCF with new density matrix
    b.scf(dm_u)
    
    transition_momentum(mol, a, b)
