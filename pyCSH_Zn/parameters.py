import numpy as np
seed = 23137
size = (4,4,4)   # Minimum (1,1,1)
Ca_Si_ratio = 2.0
W_Si_ratio  = 0.2

prefix = "CaSi"+str(Ca_Si_ratio)

N_samples = 1
make_independent = True

offset_gaussian = False
width_Ca_Si = 0.04
width_SiOH = 0.05
width_CaOH = 0.05

create =True
check = False

write_lammps = True
write_lammps_erica = False
write_lammps_cementff = True
write_vasp = False
write_siesta = False

enable_zinc = False
Zn_Si_ratio = 0.0
Zn_site_type = "Q2b_Zn"
Zn_seed = seed
Zn_charge_balance_mode = "hydroxylate_two_oxygens"
allow_unbalanced_for_debug = False
allow_hydroxylate_bridging_oxygen = False
precondition_zinc_geometry = True
target_Zn_O_distance = 1.95
write_zinc_summary = True
water_min_H_Ca = 1.6
water_min_H_Si = 1.6
water_min_H_Zn = 1.4
water_min_H_O_nonbonded = 1.2
water_min_H_H_nonbonded = 1.2
water_min_Ow_Ca = 2.2
water_min_Ow_O = 2.2
enable_water_resampling = True
max_water_resample_attempts = 100
max_structure_resample_attempts = 50
allow_lower_water_content_if_needed = False
water_hard_H_Ca = 1.3
water_hard_H_Si = 1.4
water_hard_H_Zn = 1.3
water_hard_H_O_nonbonded = 1.1
water_hard_H_H_nonbonded = 1.1
water_hard_Ow_O = 1.9
water_hard_Ow_Ca = 1.9
water_warn_H_Ca = 1.6
water_warn_H_Si = 1.6
water_warn_H_Zn = 1.4
water_warn_H_O_nonbonded = 1.2
water_warn_H_H_nonbonded = 1.2
water_warn_Ow_O = 2.2
water_warn_Ow_Ca = 2.2

orthogonal = True
shift = False
diferentiate = True
dpore = 10

guest_ions = False
substitute = np.array([["Ca1", "Zn", 5, 0.848],["Ca2", "Mn", 5, 0.848]], dtype = object) #sustituted ele, sustitute ele , sustitution %, charge
saturation = True
grid = np.array([5, 5, 5, "Cl", 1, "Na", 1], dtype = object)

# The input below allows to read a handmade brick code
# If NOT using a surface, remove "surface_separation", or set it to "surface_separation = False"

read_structure = False

shape_read = (3,3,2)
brick_code = { 
(  0,   0,   0)  :   ['<Lo', 'CU', '<R', '>L', 'CD', 'oMDR', '>R'], 
(  0,   0,   1)  :   ['<L', '<R', 'XD', 'CIU', 'oDL', '>Lo', '>R'], 
(  0,   1,   0)  :   ['<L', 'CU', '<R', 'XU', 'oUL', '>L', '>Ro'], 
(  0,   1,   1)  :   ['<L', '<R', 'XU', 'XD', 'oUL', 'oXU', '>Lo', '>R'], 
(  0,   2,   0)  :   ['<L', 'CU', '<R', 'CII', 'oDR', '>Lo', '>R'], 
(  0,   2,   1)  :   ['<Lo', 'CU', 'oMUL', '<R', 'XD', '>L', '>R'], 
(  1,   0,   0)  :   ['<L', 'SU', 'oMUL', '<R', 'CII', 'XU', 'XD', 'CID', 'CIU', 'oDL', 'oUR', 'oXU', '>L', 'SDo', 'oMDL', '>R'], 
(  1,   0,   1)  :   ['<L', '<R', 'XU', 'oDL', '>Lo', 'CD', '>R'], 
(  1,   1,   0)  :   ['<L', 'CU', '<R', '>L', 'CD', 'oMDR', '>R'], 
(  1,   1,   1)  :   ['<L', 'SUo', 'oMUL', '<R', 'CII', 'XU', 'XD', 'CID', 'CIU', 'oDL', 'oUR', 'oXU', '>L', 'SD', 'oMDL', '>R'], 
(  1,   2,   0)  :   ['<L', 'CU', '<R', 'CII', 'oDR', '>Lo', '>R'], 
(  1,   2,   1)  :   ['<Lo', 'CU', 'oMUL', '<R', 'XU', '>L', '>R'], 
(  2,   0,   0)  :   ['<Lo', '<R', 'XD', 'oUL', 'oXD', '>L', 'CD', '>R'], 
(  2,   0,   1)  :   ['<L', 'SU', 'oMUL', '<R', 'CII', 'XU', 'XD', 'CID', 'CIU', 'oDL', 'oUR', 'oXU', '>L', 'SD', 'oMDL', '>R'], 
(  2,   1,   0)  :   ['<Lo', '<R', 'XD', 'CIU', 'oDL', 'oUR', '>L', '>R'], 
(  2,   1,   1)  :   ['<Lo', '<R', 'XD', '>L', 'CD', 'oMDR', '>R'], 
(  2,   2,   0)  :   ['<L', 'CU', 'oMUL', 'oMUR', '<Ro', 'CID', '>L', '>R'], 
(  2,   2,   1)  :   ['<L', 'CU', '<R', 'CII', 'oDL', '>L', '>Ro'], 
}

water_code = { 
(  0,   0,  0)  :   ['wMDL', 'wXD', 'wUL', 'wIR2', 'wIR'], 
(  0,   0,  1)  :   ['wDR', 'wXD', 'wMDR', 'wMUR', 'w15'], 
(  0,   1,  0)  :   ['wMUR', 'wDR', 'wMUL', 'w15', 'wMDR'], 
(  0,   1,  1)  :   ['w16', 'wIR', 'w15', 'wIR2', 'wIL'], 
(  0,   2,  0)  :   ['wXD', 'wIR', 'w16', 'wMUL', 'wIR2'], 
(  0,   2,  1)  :   ['w14', 'wDR', 'wIL', 'wXD', 'wIR'], 
(  1,   0,  0)  :   ['wXD', 'w16', 'w15', 'wIL'], 
(  1,   0,  1)  :   ['wMDR', 'wIL', 'wIR2', 'wIR'], 
(  1,   1,  0)  :   ['wMUR', 'w15', 'wDR', 'w16'], 
(  1,   1,  1)  :   ['wIR2', 'w14', 'w15', 'wXD'], 
(  1,   2,  0)  :   ['wMDL', 'w16', 'wUL', 'wIR2'], 
(  1,   2,  1)  :   ['wIL', 'wXD', 'wIR', 'wMUR'], 
(  2,   0,  0)  :   ['wMDL', 'w14', 'wMUR', 'w15'], 
(  2,   0,  1)  :   ['w16', 'wIR2', 'w14', 'wIL'], 
(  2,   1,  0)  :   ['wXU', 'wMUL', 'wUL', 'w14'], 
(  2,   1,  1)  :   ['wIL', 'wMDL', 'w16', 'wXD'], 
(  2,   2,  0)  :   ['w16', 'w15', 'wXD', 'wIL'], 
(  2,   2,  1)  :   ['wMUR', 'wIR', 'wIL', 'wXD'], 
}
