import kwant
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.constants as const
import logging

# --- System Parameters & Input Data ---
Distance = 2.9 # Angstroms
Left_lead = 1
Central_region = 12
Right_lead = 1
Orbital_per_site = 1 
Fermi_energy_eV = -0.508971 

# Onsite energies (eV) for all atoms in the chain
Onsite_energies_eV = [
    -2.186082e+00, -9.748909e-01, -1.037788e+00, -9.924297e-01,
    -1.008730e+00, -9.992839e-01, -1.002512e+00, -1.002392e+00,
    -9.990355e-01, -1.008527e+00, -9.923733e-01, -1.037848e+00,
    -9.749227e-01, -2.186052e+00,
]

# Nearest-neighbor hopping parameter (eV)
Hopping_parameter_eV = [
    -1.661131e+00, -1.411491e+00, -1.421008e+00, -1.407596e+00,
    -1.417379e+00, -1.408919e+00, -1.416753e+00, -1.408925e+00,
    -1.417384e+00, -1.407599e+00, -1.421004e+00, -1.411486e+00,
    -1.661133e+00,
]

# Validate input data sizes
TOTAL_ATOMS = Left_lead + Central_region + Right_lead
if len(Onsite_energies_eV) != TOTAL_ATOMS:
    raise ValueError(f"Onsite_energies_eV length mismatch: expected {TOTAL_ATOMS}, got {len(Onsite_energies_eV)}.")
if len(Hopping_parameter_eV) != TOTAL_ATOMS - 1:
    raise ValueError(f"Hopping_parameter_eV length mismatch: expected {TOTAL_ATOMS - 1}, got {len(Hopping_parameter_eV)}.")
if Orbital_per_site != 1:
    logging.warning(f"Orbital_per_site is {Orbital_per_site}. This script is designed for 1 orbital per atom.")

# --- Physical Constants ---
G0 = 2 * const.e**2 / const.h  
KB_J_PER_K = const.k           
EV_TO_JOULE = const.e          

# --- Simulation Settings ---
if Fermi_energy_eV is None or np.isnan(Fermi_energy_eV):
    logging.error("Fermi_energy_eV is not valid (None or NaN). Please provide a valid Fermi energy.")
    raise ValueError("Invalid Fermi_energy_eV.")
else:
    FERMI_ENERGY_eV = Fermi_energy_eV
    logging.info(f"Using strictly defined Fermi Energy: {FERMI_ENERGY_eV:.4f} eV")

temp_K = 1e-5  # temp for finite-T conductance calculation (Kelvin)

# --- Fermi-Dirac Distribution Functions ---
def fermi_dirac(energy_ev, chemical_potential_ev, temp_k):
    if temp_k < 1e-10: 
        return 1.0 if energy_ev < chemical_potential_ev else 0.0 if energy_ev > chemical_potential_ev else 0.5
    kt_ev = KB_J_PER_K * temp_k / EV_TO_JOULE
    if kt_ev == 0: 
        return 1.0 if energy_ev <= chemical_potential_ev else 0.0
    exp_arg = np.clip((energy_ev - chemical_potential_ev) / kt_ev, -500, 500)
    return 1.0 / (1.0 + np.exp(exp_arg))

def fermi_dirac_derivative(energy_ev, chemical_potential_ev, temp_k):
    if temp_k < 1e-6: 
        return np.inf if abs(energy_ev - chemical_potential_ev) < 1e-9 else 0.0
    kt_ev = KB_J_PER_K * temp_k / EV_TO_JOULE
    if kt_ev == 0:
        return np.inf if abs(energy_ev - chemical_potential_ev) < 1e-9 else 0.0
    x = (energy_ev - chemical_potential_ev) / (2 * kt_ev)
    cosh_x_clipped = np.cosh(np.clip(x, -700, 700)) 
    if cosh_x_clipped == 0: return np.inf 
    return (1.0 / (4 * kt_ev)) * (1.0 / cosh_x_clipped**2)

# --- Kwant System Definition ---
def make_system(lattice_spacing=Distance, num_orbitals=Orbital_per_site):
    lat = kwant.lattice.chain(lattice_spacing, norbs=num_orbitals)
    syst_builder = kwant.Builder()

    central_kwant_sites = []
    if Central_region > 0:
        for i in range(Central_region):
            atom_idx_in_chain = Left_lead + i
            central_site_tag = i 
            site = lat(central_site_tag)
            central_kwant_sites.append(site)
            syst_builder[site] = Onsite_energies_eV[atom_idx_in_chain]
        
        for i in range(Central_region - 1):
            hopping_idx_in_chain = Left_lead + i
            syst_builder[central_kwant_sites[i], central_kwant_sites[i+1]] = Hopping_parameter_eV[hopping_idx_in_chain]

    inter_cell_hopping_eV = Hopping_parameter_eV[0] 
    logging.info(f"Lead inter-cell hopping: {inter_cell_hopping_eV:.4f} eV (from Atom 0-1)")
    lead_unit_cell_sites = 1 

    if Left_lead > 0:
        left_lead_sym = kwant.TranslationalSymmetry((-lattice_spacing * lead_unit_cell_sites,))
        left_lead_builder = kwant.Builder(left_lead_sym)
        lead_site_tag = 0 
        left_lead_builder[lat(lead_site_tag)] = Onsite_energies_eV[0] 
        left_lead_builder[lat(lead_site_tag - lead_unit_cell_sites), lat(lead_site_tag)] = inter_cell_hopping_eV
        if Central_region > 0:
            syst_builder.attach_lead(left_lead_builder, central_kwant_sites[0])
        else: 
            syst_builder.attach_lead(left_lead_builder) 

    if Right_lead > 0:
        right_lead_sym = kwant.TranslationalSymmetry((lattice_spacing * lead_unit_cell_sites,))
        right_lead_builder = kwant.Builder(right_lead_sym)
        lead_site_tag = 0
        first_atom_right_lead_idx = Left_lead + Central_region 
        right_lead_builder[lat(lead_site_tag)] = Onsite_energies_eV[first_atom_right_lead_idx]
        right_lead_builder[lat(lead_site_tag + lead_unit_cell_sites), lat(lead_site_tag)] = inter_cell_hopping_eV
        if Central_region > 0:
            syst_builder.attach_lead(right_lead_builder, central_kwant_sites[Central_region-1])
        elif Left_lead == 0: 
             syst_builder.attach_lead(right_lead_builder)
    return syst_builder.finalized()

# --- DOS Calculation for Central Region (Edited for Flexibility) ---
def calculate_total_dos_central_region(
    finalized_system,
    energies_ev,
    num_central_atoms_defined,
    Orbital_per_site
):
    if num_central_atoms_defined == 0:
        logging.info("No explicit central region defined (0 atoms). DOS calculation skipped.")
        return np.full_like(energies_ev, np.nan)

    # Get the actual number of sites in the finalized scattering region
    actual_scattering_sites = len(finalized_system.sites)

    # Check for mismatch but only issue a warning instead of failing later
    if actual_scattering_sites != num_central_atoms_defined:
        logging.warning(
            f"Central region site count mismatch: Expected {num_central_atoms_defined} (defined), "
            f"got {actual_scattering_sites} in finalized Kwant scattering region. "
            f"Proceeding with DOS calculation for the {actual_scattering_sites} actual sites."
        )

    if Orbital_per_site <= 0:
        logging.error(f"Invalid Orbital_per_site ({Orbital_per_site}). Cannot calculate DOS.")
        return np.full_like(energies_ev, np.nan)

    dos_values = np.full_like(energies_ev, np.nan)

    expected_ldos_array_size = actual_scattering_sites * Orbital_per_site

    for i, energy in enumerate(energies_ev):
        try:
            ldos_contributions = kwant.ldos(finalized_system, energy)

            # This check now serves as a sanity check on the output of kwant.ldos
            if ldos_contributions.size == expected_ldos_array_size:
                dos_values[i] = np.sum(ldos_contributions)
            else:
                # This warning is now more serious, as it indicates an unexpected
                # output size from the Kwant calculation itself.
                logging.warning(
                    f"LDOS array size issue at E={energy:.4f} eV. "
                    f"Expected {expected_ldos_array_size} for the actual finalized region, "
                    f"got {ldos_contributions.size}. DOS set to NaN."
                )
        except RuntimeError as e:
            logging.warning(f"Kwant LDOS calculation error at E={energy:.4f} eV: {e}. DOS set to NaN.")
        except Exception as e:
            logging.error(f"Unexpected error in LDOS calc at E={energy:.4f} eV: {e}. DOS set to NaN.")

    return dos_values

# --- Midpoint Integration Function ---
def midpoint_integration(func, a, b, n_intervals, args=()):
    if n_intervals <= 0:
        logging.error("Number of intervals for midpoint integration must be positive.")
        return np.nan
    if a >=b: 
        if a == b: return 0.0
        logging.warning(f"Midpoint integration called with a >= b ({a}, {b}). Result might be unexpected.")

    h = (b - a) / n_intervals
    integral_sum = 0.0
    for i in range(n_intervals):
        mid_point = a + (i + 0.5) * h
        try:
            integral_sum += func(mid_point, *args)
        except Exception as e:
            logging.error(f"Error evaluating integrand at midpoint {mid_point}: {e}")
            return np.nan 
            
    return integral_sum * h

# --- Conductance Calculation ---
def compute_conductance_metrics(
    fermi_energy_ev, temp_k,
    energy_array_ev, transmission_array,
    conductance,
    min_integration_energy_ev, max_integration_energy_ev,
    num_integration_intervals=500 
    ):
    conductance_T0_S = np.nan
    conductance_finite_T_S = np.nan

    valid_energies_idx = ~np.isnan(transmission_array)
    valid_energies = energy_array_ev[valid_energies_idx]
    valid_transmission = transmission_array[valid_energies_idx]

    if len(valid_transmission) <= 1: 
        logging.warning("Not enough valid transmission points to calculate conductance.")
        return conductance_T0_S, conductance_finite_T_S

    conductance_T0_S = np.interp(fermi_energy_ev, valid_energies, valid_transmission) * conductance
    
    if temp_k > 1e-6:
        transmission_interpolator = interp1d(
            valid_energies, valid_transmission, 
            bounds_error=False, fill_value=0.0 
        )
        
        def integrand(energy_ev_arg, mu_ev_arg, temp_k_arg):
            return transmission_interpolator(energy_ev_arg) * fermi_dirac_derivative(energy_ev_arg, mu_ev_arg, temp_k_arg)

        kt_ev = KB_J_PER_K * temp_k / EV_TO_JOULE
        integration_min = max(min_integration_energy_ev, fermi_energy_ev - 20 * kt_ev if kt_ev > 0 else fermi_energy_ev - 1.5)
        integration_max = min(max_integration_energy_ev, fermi_energy_ev + 20 * kt_ev if kt_ev > 0 else fermi_energy_ev + 1.5)
        
        if integration_min < integration_max:
            try:
                integral_val = midpoint_integration(
                    integrand,
                    integration_min, integration_max,
                    num_integration_intervals, 
                    args=(fermi_energy_ev, temp_k) 
                )
                if not np.isnan(integral_val):
                    conductance_finite_T_S = conductance * integral_val
            except Exception as e: 
                logging.warning(f"Numerical integration (midpoint) for conductance failed: {e}")
        else:
            logging.info("Finite-T conductance integration range is invalid (min >= max).")
    else: 
        conductance_finite_T_S = conductance_T0_S
        
    return conductance_T0_S, conductance_finite_T_S

# --- Main Execution ---
def main():
    if Central_region == 0 and (Left_lead == 0 or Right_lead == 0):
        logging.error("System requires at least two leads if the central region has zero atoms.")
        plt.show() 
        return
        
    finalized_syst = make_system() 
    if not finalized_syst.leads:
        logging.error("No leads attached to the system. Transport calculation aborted.")
        plt.show()
        return

    energy_min_ev = -5.0
    energy_max_ev = 5.0
    energies_ev = np.linspace(energy_min_ev, energy_max_ev, 500)

    logging.info(f"Calculating T(E), DOS(E) for E in [{energy_min_ev:.2f}, {energy_max_ev:.2f}] eV. E_F={FERMI_ENERGY_eV:.4f} eV.")

    transmission_values = []
    for energy in energies_ev:
        try:
            smatrix = kwant.smatrix(finalized_syst, energy)
            T_val = smatrix.transmission(1, 0) if len(finalized_syst.leads) >= 2 else 0.0
            transmission_values.append(T_val)
        except Exception as e:
            logging.warning(f"Transmission calculation error at E={energy:.4f} eV: {e}")
            transmission_values.append(np.nan)
    transmission_values = np.array(transmission_values)
            
    dos_values = calculate_total_dos_central_region(
        finalized_syst, energies_ev, Central_region, Orbital_per_site
    )

    G_at_Ef, G_finite_temp_at_Ef = compute_conductance_metrics(
        FERMI_ENERGY_eV, temp_K, energies_ev, transmission_values, 
        G0, energy_min_ev, energy_max_ev,
        num_integration_intervals=500 
    )
    if len(finalized_syst.leads) < 2: 
        G_at_Ef, G_finite_temp_at_Ef = np.nan, np.nan
        logging.info("Skipping conductance value calculation: Less than 2 leads.")

    logging.info(f"G(T=0K, E_F={FERMI_ENERGY_eV:.4f}eV): {G_at_Ef:.4e} S")
    logging.info(f"G(T={temp_K}K, E_F={FERMI_ENERGY_eV:.4f}eV): {G_finite_temp_at_Ef:.4e} S")

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True) 
    plot_title_suffix = f"(Li Chain '2s', $E_F={FERMI_ENERGY_eV:.4f}$ eV)"
    
    axs[0].plot(energies_ev, transmission_values, color='dodgerblue')
    axs[0].set_title(f'Transmission T(E) {plot_title_suffix}')
    axs[0].set_ylabel('T(E)')
    axs[0].axvline(FERMI_ENERGY_eV, color='red', linestyle='--', label='$E_F$')
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.7)
    
    axs[1].plot(energies_ev, dos_values, color='forestgreen')
    axs[1].set_title(f'DOS (Central Region) {plot_title_suffix}')
    axs[1].set_ylabel('DOS (states/eV)')
    axs[1].axvline(FERMI_ENERGY_eV, color='red', linestyle='--', label='$E_F$')
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.7)

    conductance_vs_energy = G0 * transmission_values
    axs[2].plot(energies_ev, conductance_vs_energy, color='darkorange', label='G(E) = $G_0 \\times T(E)$')
    axs[2].set_title(f'Conductance G(E) {plot_title_suffix}')
    axs[2].set_ylabel('Conductance (S)')
    axs[2].axvline(FERMI_ENERGY_eV, color='red', linestyle='--', label='$E_F$')
    
    if not np.isnan(G_at_Ef):
        axs[2].axhline(G_at_Ef, color='blueviolet', linestyle=':', 
                       label=f'G($E_F$, T=0K) = {G_at_Ef:.2e} S')
    if not np.isnan(G_finite_temp_at_Ef):
        axs[2].axhline(G_finite_temp_at_Ef, color='deeppink', linestyle='-.', 
                       label=f'G($E_F$, T={temp_K}K) = {G_finite_temp_at_Ef:.2e} S')
                       
    axs[2].legend()
    axs[2].grid(True, linestyle=':', alpha=0.7)
    axs[2].set_xlabel('Energy (eV)')

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    main()
