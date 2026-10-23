"""
Modified from github below:
https://github.com/Jhsmit/HDXrate/tree/master
"""

"""
This script is a python implementation of the Englander xls sheets for intrinsic rates
Copyright (C) 2020 Jochem Smit

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import numpy as np

R = 1.987 # gas constant, cal K-1 mol-1

# Activation energies (cal/mol)
E_act = {
    'acid': 14000.,
    'base': 17000.,
    'water': 19000.,
    'D': 1000.,
    'E': 1083.,
    'H': 7500.
}

D_E_act = {
    'D_HD' : 1000.,
    'D_DH' : 1000. - 40,
    'D_HH' : 1000. - 40
}

def get_side_chain_dictionary(temperature, pH, k_reference):
    """
    Returns a dictionary with inductive effects of side chains on H/D exchange rates.

    Values are from [1]_, [2]_, [3]_, as described in [4]_

    Parameters
    ----------
    temperature: :obj:`float`
        Temperature in Kelvin.
    pH: :obj:`float`
        pH/pD value.
    k_reference: :obj:`dict`
        Dictionary of references values for amino acids D, E, H

    Returns
    -------

    constants: :obj:`dict`
        Dictionary of side chain modifiers. Values are: (acid_lambda, acid_rho, base_lambda, base_rho)

    References
    ----------

    .. [1] Bai, Y., Milne, J. S., Mayne, L. & Englander, S. W. Primary structure effects on peptide group hydrogen
       exchange. Proteins: Structure, Function, and Bioinformatics 17, 75–86 (1993).
    .. [2] Connelly, G. P., Bai, Y., Jeng, M.-F. & Englander, S. W. Isotope effects in peptide group hydrogen
       exchange. Proteins 17, 87–92 (1993).
    .. [3] Mori, S., Zijl, P. C. M. van & Shortle, D. Measurement of water–amide proton exchange rates in the
       denatured state of staphylococcal nuclease by a magnetization transfer technique. Proteins: Structure,
       Function, and Bioinformatics 28, 325–332 (1997).
    .. [4] Nguyen, D., Mayne, L., Phillips, M. C. & Walter Englander, S. Reference Parameters for Protein Hydrogen
       Exchange Rates. J. Am. Soc. Mass Spectrom. 29, 1936–1939 (2018).
    """

    names = ['name', 'short_name', 'acid_lambda', 'acid_rho', 'base_lambda', 'base_rho']

    # sidechain parameters Nguyen 2018
    side_chain_array = np.array([('ALANINE', 'A',  0.  ,  0.  ,  0.  ,  0.  ),
       ('CYSTEIN', 'C', -0.54, -0.46,  0.62,  0.55),
       ('CYSTINE', 'C2', -0.74, -0.58,  0.55,  0.46),
       ('ASP COO-', 'D0',  0.9 ,  0.58,  0.1 , -0.18),
       ('ASP COOH', 'D+', -0.9 , -0.12,  0.69,  0.6 ),
       ('GLU COO-', 'E0', -0.9 ,  0.31, -0.11, -0.15),
       ('GLU COOH', 'E+', -0.6 , -0.27,  0.24,  0.39),
       ('PHENYLALANINE', 'F', -0.52, -0.43, -0.24,  0.06),
       ('GLYCINE', 'G', -0.22,  0.22, -0.03,  0.17),
       ('HIS N', 'H0',  0.  ,  0.  , -0.1 ,  0.14),
       ('HIS+', 'H+', -0.8 , -0.51,  0.8 ,  0.83),
       ('ISOLEUCINE', 'I', -0.91, -0.59, -0.73, -0.23),
       ('LYSINE', 'K', -0.56, -0.29, -0.04,  0.12),
       ('LEUCINE', 'L', -0.57, -0.13, -0.58, -0.21),
       ('METHIONINE', 'M', -0.64, -0.28, -0.01,  0.11),
       ('ASPARAGINE', 'N', -0.58, -0.13,  0.49,  0.32),
       ('PROLINE', 'P',   0., -0.19,   0., -0.24),
       ('PRO cis', 'Pc',   0., -0.85,   0.,  0.6 ),
       ('GLUTAMINE', 'Q', -0.47, -0.27,  0.06,  0.2 ),
       ('ARGININE', 'R', -0.59, -0.32,  0.08,  0.22),
       ('SERINE', 'S', -0.44, -0.39,  0.37,  0.3 ),
       ('THREONINE', 'T', -0.79, -0.47, -0.07,  0.2 ),
       ('VALINE', 'V', -0.74, -0.3 , -0.7 , -0.14),
       ('TRYPTOPHANE', 'W', -0.4 , -0.44, -0.41, -0.11),
       ('TYROSINE', 'Y', -0.41, -0.37, -0.27,  0.05),
       ('N-Term', 'NT',   0., -1.32,   0.,  1.62),
       ('C-Term', 'CT',   0.,   0., -1.8 ,   0.)]) # C-Term lambda acid value calculated based on pH and pKa

    side_chain_dict = {elem[1]: np.array(list(elem)[2:]) for elem in side_chain_array} # elem[1] = short name
    for residue in ['D', 'E', 'H']: # residues D, E, H are calculated based on pH and pKa
        k_corrected = -np.log10(10**-k_reference[residue] * np.exp(-E_act[residue] * (1 / temperature - 1 / 278) / R)) # Check correct reference temperature

        deprotenated = side_chain_dict[residue + '0']
        protenated = side_chain_dict[residue + '+']

        # Ensure protenated and deprotenated values are float arrays before performing arithmetic
        protenated = np.array(protenated, dtype=float)
        deprotenated = np.array(deprotenated, dtype=float)

        values = np.log10(np.divide(10 ** (protenated - pH) + 10 ** (deprotenated - k_corrected),
                                10 ** -k_corrected + 10 ** -pH))

        side_chain_dict[residue] = values
        if residue == 'E':
            side_chain_dict['CT'][0] = np.log10(np.divide(10 ** (0.05 - pH) + 10 ** (0.96 - k_corrected),
                                               10 ** -k_corrected + 10 ** -pH))

    return side_chain_dict


def seq_to_list(seq):
    if 'Pc' not in seq and 'C2' not in seq:
        return list(seq)
    else:
        interm = list(seq)
        final = []
        for ele in interm:
            if ele != 'c' and ele != '2':
                final.append(ele)
            else:
                final[-1]=final[-1]+ele

        return list(final)


def Nguyen2018(sequence, temperature, pH, reference='poly', exchange_type='HD',
                        hd_percentage=100., ph_correction=False, wildcard='X'):
    """
    Calculated intrisic rates of exchange for amide hydrogens in proteins.

    Calculations are based on  [1]_, [2]_, [3]_ and [4]_.


    Parameters
    ----------
    sequence: iterable object
        Input sequence in single-letter amino acid codes. Use 'Pc' for cis Proline, 'C2' for Cystine (disulfide)
    temperature: :obj:`float`
        Temperature of the exchange reaction in Kelvin.
    pH: :obj:`float`
        pH corrected for isotope effects if relevant (when D2O present). pH corr = pH read + 0.4 in 100 % D2O
    reference: :obj:`str`
        Use `poly` or `oligo` reference data.
    exchange_type: :obj:`str`
        The type of exchange. Options are `HD`, `DH` or `HH`.
    d_percentage: :obj:`float`
        Percentage of Deuterium in the reaction solution. Used for pH/pD correction.
    ph_correction: :obj:`bool`
        Whether or not to correct the supplied `pH` value to pD. `DH` and `HH` exchange pH is not corrected.
    wildcard: :obj:`str`:
        Wildcard to use for unknown amino acids in the sequence. Amino acids with the wildcard and amino acids to the
        right (C-term) of wildcard residues return 0. as rate.

    Returns
    -------
    rates : :class:`~numpy.ndarray`
        Array with exchange rates in units of per second.

    References
    ----------

    .. [1] Bai, Y., Milne, J. S., Mayne, L. & Englander, S. W. Primary structure effects on peptide group hydrogen
       exchange. Proteins: Structure, Function, and Bioinformatics 17, 75–86 (1993).
    .. [2] Connelly, G. P., Bai, Y., Jeng, M.-F. & Englander, S. W. Isotope effects in peptide group hydrogen
       exchange. Proteins 17, 87–92 (1993).
    .. [3] Mori, S., Zijl, P. C. M. van & Shortle, D. Measurement of water–amide proton exchange rates in the
       denatured state of staphylococcal nuclease by a magnetization transfer technique. Proteins: Structure,
       Function, and Bioinformatics 28, 325–332 (1997).
    .. [4] Nguyen, D., Mayne, L., Phillips, M. C. & Walter Englander, S. Reference Parameters for Protein Hydrogen
       Exchange Rates. J. Am. Soc. Mass Spectrom. 29, 1936–1939 (2018).
    """

    if len(sequence) <3:
        raise ValueError('Sequence needs a minimum length of 3')
    if exchange_type not in ['HD', 'DH', 'HH']:
        raise ValueError(f"Unsupported exchange type '{exchange_type}'")

    if exchange_type == 'HD':
        exponents = np.array([1.62, 10.18, -1.5])
        pD = pH
        pKD = (15.05*hd_percentage + 14.17*(100-hd_percentage))/100
        k_reference = {'D': 4.48, 'E': 4.93, 'H': 7.42}  # HD
        E_act['D'] = D_E_act['D_HD']
    elif exchange_type == 'DH':
        exponents = np.array([1.4, 10., -1.6])
        pD = pH
        pKD = (14.17*hd_percentage + 15.05*(100-hd_percentage))/100
        E_act['D'] = D_E_act['D_DH']
        k_reference = {'D': 3.87, 'E': 4.33, 'H': 7.0}  #DH
    elif exchange_type == 'HH':
        exponents = np.array([1.39, 10.08, -1.6])
        pD = pH
        pKD = (14.17*hd_percentage + 15.05*(100-hd_percentage))/100
        E_act['D'] = D_E_act['D_HH']
        k_reference = {'D': 3.88, 'E': 4.35, 'H': 7.11}  #HH

    conc_D = 10. ** -pD
    conc_OD = 10. ** (pD - pKD)

    k_values = (10 ** exponents) / 60 # convert min-1 to s-1
    oligo_factors = [2.34, 1.35, 1.585]
    if reference == 'poly':
        k_acid_ref, k_base_ref, k_water_ref = k_values
    elif reference == 'oligo':
        k_acid_ref, k_base_ref, k_water_ref = k_values * oligo_factors
    else:
        raise ValueError("Value for 'reference' mush be either 'poly' or 'oligo'")

    sequence = seq_to_list(sequence)
    sequence.insert(0, 'NT')
    sequence.append('CT')

    # Rates without inductive effects from neighbours, corrected for temperature
    k_acid = k_acid_ref * np.exp(-E_act['acid'] * (1 / temperature - 1 / 293) / R)
    k_base = k_base_ref * np.exp(-E_act['base'] * (1 / temperature - 1 / 293) / R)
    k_water = k_water_ref * np.exp(-E_act['water'] * (1 / temperature - 1 / 293) / R)

    side_chain_dict = get_side_chain_dictionary(temperature, pD, k_reference)

    k_int = []
    for i, residue in enumerate(sequence):
        if residue == 'NT':
            continue
        elif i == 1:  # First residue
            k_int.append(np.inf)
            continue

        next_residue = sequence[i + 1]
        prev_residue = sequence[i - 1]
        # Proline or unknown residues are set to zero rate
        if residue in ['P', 'Pc'] or wildcard in [prev_residue, residue]:
            k_int.append(0.)
            if next_residue == 'CT':
                break
            else:
                continue

        # Format is left_acid, right_acid, left_base, right_base
        _, prev_rho_acid, _, prev_rho_base = side_chain_dict[prev_residue]
        curr_lambda_acid, _, curr_lambda_base, _ = side_chain_dict[residue]

        if next_residue == 'CT':
            cterm_acid = side_chain_dict['CT'][0]
            cterm_base = side_chain_dict['CT'][2]

            Fa = 10 ** (float(prev_rho_acid) + float(curr_lambda_acid) + float(cterm_acid))
            Fb = 10 ** (float(curr_lambda_base) + float(prev_rho_base) + float(cterm_base))
        elif i == 2:  # Second residue in the chain (starts at 1)
            nterm_acid = side_chain_dict['NT'][1]
            nterm_base = side_chain_dict['NT'][3]

            Fa = 10 ** (float(prev_rho_acid) + float(curr_lambda_acid) + float(nterm_acid))
            Fb = 10 ** (float(curr_lambda_base) + float(prev_rho_base) + float(nterm_base))
        else:
            Fa = 10 ** (float(prev_rho_acid) + float(curr_lambda_acid))
            Fb = 10 ** (float(curr_lambda_base) + float(prev_rho_base))

        k_total_acid = Fa*k_acid*conc_D
        k_total_base = Fb*k_base*conc_OD
        k_total_water = Fb*k_water

        k_int.append(k_total_acid + k_total_base + k_total_water)

        if next_residue == 'CT':
            break

    return np.array(k_int)

def Stofella2025(sequence, temperature, pH, reference='poly', exchange_type='HD',
                        hd_percentage=100., ph_correction=False, wildcard='X'):
    """
    Calculated intrisic rates of exchange for amide hydrogens in proteins.

    Calculations are based on  [1]_, [2]_, [3]_ and [4]_.


    Parameters
    ----------
    sequence: iterable object
        Input sequence in single-letter amino acid codes. Use 'Pc' for cis Proline, 'C2' for Cystine (disulfide)
    temperature: :obj:`float`
        Temperature of the exchange reaction in Kelvin.
    pH: :obj:`float`
        pH corrected for isotope effects if relevant (when D2O present). pH corr = pH read + 0.4 in 100 % D2O
    reference: :obj:`str`
        Use `poly` or `oligo` reference data.
    exchange_type: :obj:`str`
        The type of exchange. Options are `HD`, `DH` or `HH`.
    d_percentage: :obj:`float`
        Percentage of Deuterium in the reaction solution. Used for pH/pD correction.
    ph_correction: :obj:`bool`
        Whether or not to correct the supplied `pH` value to pD. `DH` and `HH` exchange pH is not corrected.
    wildcard: :obj:`str`:
        Wildcard to use for unknown amino acids in the sequence. Amino acids with the wildcard and amino acids to the
        right (C-term) of wildcard residues return 0. as rate.

    Returns
    -------
    rates : :class:`~numpy.ndarray`
        Array with exchange rates in units of per second.

    References
    ----------

    .. [1] Bai, Y., Milne, J. S., Mayne, L. & Englander, S. W. Primary structure effects on peptide group hydrogen
       exchange. Proteins: Structure, Function, and Bioinformatics 17, 75–86 (1993).
    .. [2] Connelly, G. P., Bai, Y., Jeng, M.-F. & Englander, S. W. Isotope effects in peptide group hydrogen
       exchange. Proteins 17, 87–92 (1993).
    .. [3] Mori, S., Zijl, P. C. M. van & Shortle, D. Measurement of water–amide proton exchange rates in the
       denatured state of staphylococcal nuclease by a magnetization transfer technique. Proteins: Structure,
       Function, and Bioinformatics 28, 325–332 (1997).
    .. [4] Nguyen, D., Mayne, L., Phillips, M. C. & Walter Englander, S. Reference Parameters for Protein Hydrogen
       Exchange Rates. J. Am. Soc. Mass Spectrom. 29, 1936–1939 (2018).
    """

    if len(sequence) <3:
        raise ValueError('Sequence needs a minimum length of 3')
    if exchange_type not in ['HD', 'DH', 'HH']:
        raise ValueError(f"Unsupported exchange type '{exchange_type}'")
    
    # the acid, base, water rate constants correspond to the 3-Ala reference from Connelly 1993 (HD exchange)
    # for DH and HH exchange, the rate constants were empirically increased by the same amount as the HD constants
    # (relative to the Bai 1993 parameters)
    # acid rate constant increase (log10): 0.42
    # base rate constant increase (log10): 0.31
    # water rate constant increase (log10): 0.00

    if exchange_type == 'HD':
        exponents = np.array([2.04, 10.36, -1.5])
        pD = pH
        pKD = (15.05*hd_percentage + 14.17*(100-hd_percentage))/100
        k_reference = {'D': 4.48, 'E': 4.93, 'H': 7.42}  # HD
        E_act['D'] = D_E_act['D_HD']
    elif exchange_type == 'DH':
        exponents = np.array([1.82, 10.31, -1.6])
        pD = pH
        pKD = (14.17*hd_percentage + 15.05*(100-hd_percentage))/100
        E_act['D'] = D_E_act['D_DH']
        k_reference = {'D': 3.87, 'E': 4.33, 'H': 7.0}  #DH
    elif exchange_type == 'HH':
        exponents = np.array([1.81, 10.39, -1.6])
        pD = pH
        pKD = (14.17*hd_percentage + 15.05*(100-hd_percentage))/100
        E_act['D'] = D_E_act['D_HH']
        k_reference = {'D': 3.88, 'E': 4.35, 'H': 7.11}  #HH

    conc_D = 10. ** -pD
    conc_OD = 10. ** (pD - pKD)

    k_values = (10 ** exponents) / 60 # convert min-1 to s-1
    oligo_factors = [2.34, 1.35, 1.585]
    if reference == 'poly':
        k_acid_ref, k_base_ref, k_water_ref = k_values
    elif reference == 'oligo':
        k_acid_ref, k_base_ref, k_water_ref = k_values * oligo_factors
    else:
        raise ValueError("Value for 'reference' mush be either 'poly' or 'oligo'")

    sequence = seq_to_list(sequence)
    sequence.insert(0, 'NT')
    sequence.append('CT')

    # Rates without inductive effects from neighbours, corrected for temperature
    k_acid = k_acid_ref * np.exp(-E_act['acid'] * (1 / temperature - 1 / 293) / R)
    k_base = k_base_ref * np.exp(-E_act['base'] * (1 / temperature - 1 / 293) / R)
    k_water = k_water_ref * np.exp(-E_act['water'] * (1 / temperature - 1 / 293) / R)

    side_chain_dict = get_side_chain_dictionary(temperature, pD, k_reference)

    k_int = []
    for i, residue in enumerate(sequence):
        if residue == 'NT':
            continue
        elif i == 1:  # First residue
            k_int.append(np.inf)
            continue

        next_residue = sequence[i + 1]
        prev_residue = sequence[i - 1]
        # Proline or unknown residues are set to zero rate
        if residue in ['P', 'Pc'] or wildcard in [prev_residue, residue]:
            k_int.append(0.)
            if next_residue == 'CT':
                break
            else:
                continue

        # Format is left_acid, right_acid, left_base, right_base
        _, prev_rho_acid, _, prev_rho_base = side_chain_dict[prev_residue]
        curr_lambda_acid, _, curr_lambda_base, _ = side_chain_dict[residue]

        if next_residue == 'CT':
            cterm_acid = side_chain_dict['CT'][0]
            cterm_base = side_chain_dict['CT'][2]

            Fa = 10 ** (float(prev_rho_acid) + float(curr_lambda_acid) + float(cterm_acid))
            Fb = 10 ** (float(curr_lambda_base) + float(prev_rho_base) + float(cterm_base))
        elif i == 2:  # Second residue in the chain (starts at 1)
            nterm_acid = side_chain_dict['NT'][1]
            nterm_base = side_chain_dict['NT'][3]

            Fa = 10 ** (float(prev_rho_acid) + float(curr_lambda_acid) + float(nterm_acid))
            Fb = 10 ** (float(curr_lambda_base) + float(prev_rho_base) + float(nterm_base))
        else:
            Fa = 10 ** (float(prev_rho_acid) + float(curr_lambda_acid))
            Fb = 10 ** (float(curr_lambda_base) + float(prev_rho_base))

        k_total_acid = Fa*k_acid*conc_D
        k_total_base = Fb*k_base*conc_OD
        k_total_water = Fb*k_water

        k_int.append(k_total_acid + k_total_base + k_total_water)

        if next_residue == 'CT':
            break

    return np.array(k_int)
