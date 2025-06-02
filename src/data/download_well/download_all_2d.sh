python download_well_2d.py --dataset_name turbulent_radiative_layer_2D --split_name train --out_name turbulent_radiative_layer_2D_train
python download_well_2d.py --dataset_name turbulent_radiative_layer_2D --split_name valid --out_name turbulent_radiative_layer_2D_valid
python download_well_2d.py --dataset_name turbulent_radiative_layer_2D --split_name test --out_name turbulent_radiative_layer_2D_test

python download_well_2d.py --dataset_name active_matter --split_name train --out_name active_matter_train
python download_well_2d.py --dataset_name active_matter --split_name valid --out_name active_matter_valid
python download_well_2d.py --dataset_name active_matter --split_name test --out_name active_matter_test

python download_well_2d.py --dataset_name viscoelastic_instability --split_name train --out_name viscoelastic_instability_train
python download_well_2d.py --dataset_name viscoelastic_instability --split_name valid --out_name viscoelastic_instability_valid
python download_well_2d.py --dataset_name viscoelastic_instability --split_name test --out_name viscoelastic_instability_test

python download_well_2d.py --dataset_name helmholtz_staircase --split_name train --out_name helmholtz_staircase_train
python download_well_2d.py --dataset_name helmholtz_staircase --split_name valid --out_name helmholtz_staircase_valid
python download_well_2d.py --dataset_name helmholtz_staircase --split_name test --out_name helmholtz_staircase_test

python download_well_2d.py --dataset_name rayleigh_benard --split_name train --out_name rayleigh_benard_train --reduce_fraction 0.2 --reduce_seed 1
python download_well_2d.py --dataset_name rayleigh_benard --split_name valid --out_name rayleigh_benard_valid --reduce_fraction 0.2 --reduce_seed 2
python download_well_2d.py --dataset_name rayleigh_benard --split_name test --out_name rayleigh_benard_test --reduce_fraction 0.2 --reduce_seed 3

python download_well_2d.py --dataset_name shear_flow --split_name train --out_name shear_flow_train --reduce_fraction 0.1 --reduce_seed 4
python download_well_2d.py --dataset_name shear_flow --split_name valid --out_name shear_flow_valid --reduce_fraction 0.1 --reduce_seed 5
python download_well_2d.py --dataset_name shear_flow --split_name test --out_name shear_flow_test --reduce_fraction 0.1 --reduce_seed 6