"""
mapping.py - Map a Hamiltonian to a LinearCombinaisonPauliString

Copyright 2020-2021 Maxime Dion <maxime.dion@usherbrooke.ca>
This file has been modified by <Your,Name> during the
QSciTech-QuantumBC virtual workshop on gate-based quantum computing.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pauli_string import PauliString, LinearCombinaisonPauliString
import numpy as np
from typing import Union,Tuple,List


class Mapping:

    def fermionic_creation_annihilation_operators(self, n_qubits: int) -> Tuple[List[LinearCombinaisonPauliString],List[LinearCombinaisonPauliString]]:
        raise NotImplementedError("abstract base class implementation")
        
    def fermionic_hamiltonian_to_qubit_hamiltonian(self, fermionic_hamiltonian)->LinearCombinaisonPauliString:
        """
        Do the mapping of a FermionicHamiltonian. First generates the LCPS representation of the creation/annihilation
        operators for the specific mapping. Uses the 'to_pauli_string_linear_combinaison' of the FermionicHamiltonian
        to generate the complete LCPS.

        Args:
            fermionic_hamiltonian (FermionicHamiltonian): A FermionicHamiltonian that provided a 
                'to_pauli_string_linear_combinaison' method.

        Returns:
            LinearCombinaisonPauliString: The LCPS representing the FermionicHamiltonian
        """

        creation_operators, annihilation_operators = self.fermionic_creation_annihilation_operators(fermionic_hamiltonian.number_of_orbitals())
        qubit_hamiltonian = fermionic_hamiltonian.to_linear_combinaison_pauli_string(creation_operators, annihilation_operators)
        return qubit_hamiltonian


class JordanWigner(Mapping):
    def __init__(self):
        """
        The Jordan-Wigner mapping
        """

        self.name = 'jordan-wigner'

    def fermionic_creation_annihilation_operators(self, n_qubits: int) -> Tuple[List[LinearCombinaisonPauliString],List[LinearCombinaisonPauliString]]:
        """
        Build the LCPS reprensetations for the creation/annihilation operator for each qubit following 
        Jordan-Wigner mapping.

        Args:
            n_qubits (int): The number of orbitals to be mapped to the same number of qubits.

        Returns:
            list<LinearCombinaisonPauliString>, list<LinearCombinaisonPauliString>: Lists of the creation/annihilation
                operators for each orbital in the form of LinearCombinaisonPauliString.
        """

        creation_operators = list()
        annihilation_operators = list()
        
        
        ################################################################################################################
        # YOUR CODE HERE
        # TO COMPLETE (after lecture on mapping)
        # This is a large piece of the puzzle
        ps1 = [*'IIIX']
        ps2 = [*'IIIY']
        for i in range(n_qubits):
            pauli_string_1 = PauliString.from_str(ps1)
            pauli_string_2 = PauliString.from_str(ps2)
            pauli_strings = np.array([pauli_string_1,pauli_string_2],dtype = PauliString)
            coefs = np.array([0.5,0.5j],dtype = complex)
            lcps = LinearCombinaisonPauliString(coefs,pauli_strings)
            annihilation_operators.append(lcps)
            coefs = np.array([0.5,-0.5j],dtype = complex)
            lcps = LinearCombinaisonPauliString(coefs,pauli_strings)
            creation_operators.append(lcps)
            del ps1[0]
            ps1.append('Z')
            del ps2[0]
            ps2.append('Z')
        ################################################################################################################

        return creation_operators, annihilation_operators


class Parity(Mapping):
    def __init__(self):
        """
        The Parity mapping
        """

        self.name = 'parity'

    def fermionic_creation_annihilation_operators(self, n_qubits: int) -> Tuple[List[LinearCombinaisonPauliString],List[LinearCombinaisonPauliString]]:
        """
        Build the LCPS reprensetations for the creation/annihilation operator for each qubit following 
        Parity mapping.

        Args:
            n_qubits (int): The number of orbtials to be mapped to the same number of qubits.

        Returns:
            list<LinearCombinaisonPauliString>, list<LinearCombinaisonPauliString>: Lists of the creation/annihilation
                operators for each orbital in the form of LinearCombinaisonPauliString
        """

        creation_operators = list()
        annihilation_operators = list()
        
        ################################################################################################################
        # YOUR CODE HERE
        # OPTIONAL
        ################################################################################################################

        return creation_operators, annihilation_operators
