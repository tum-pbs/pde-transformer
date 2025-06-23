python simulation.py --dimension 2 --pde adv --out_name adv --num_sims 60 --gpu_id 2
python simulation.py --dimension 2 --pde diff --out_name diff --num_sims 60 --gpu_id 2
python simulation.py --dimension 2 --pde adv_diff --out_name adv_diff --num_sims 60 --gpu_id 2
python simulation.py --dimension 2 --pde disp --out_name disp --num_sims 60 --gpu_id 2
python simulation.py --dimension 2 --pde hyp --out_name hyp --num_sims 60 --gpu_id 2
python simulation.py --dimension 2 --pde burgers --out_name burgers --num_sims 60 --gpu_id 2
python simulation.py --dimension 2 --pde kdv --out_name kdv --num_sims 60 --gpu_id 2
python simulation.py --dimension 2 --pde ks --out_name ks --num_sims 60 --gpu_id 2
python simulation.py --dimension 2 --pde fisher --out_name fisher --num_sims 60 --gpu_id 2
python simulation.py --dimension 2 --pde sh --out_name sh --num_sims 60 --gpu_id 2
python simulation.py --dimension 2 --pde gs_alpha --out_name gs_alpha --num_sims 10 --gpu_id 2
python simulation.py --dimension 2 --pde gs_beta --out_name gs_beta --num_sims 10 --gpu_id 2
python simulation.py --dimension 2 --pde gs_gamma --out_name gs_gamma --num_sims 10 --gpu_id 2
python simulation.py --dimension 2 --pde gs_delta --out_name gs_delta --num_sims 10 --gpu_id 2
python simulation.py --dimension 2 --pde gs_epsilon --out_name gs_epsilon --num_sims 10 --gpu_id 2
python simulation.py --dimension 2 --pde gs_theta --out_name gs_theta --num_sims 10 --gpu_id 2
python simulation.py --dimension 2 --pde gs_iota --out_name gs_iota --num_sims 10 --gpu_id 2
python simulation.py --dimension 2 --pde gs_kappa --out_name gs_kappa --num_sims 10 --gpu_id 2
python simulation.py --dimension 2 --pde decay_turb --out_name decay_turb --num_sims 60 --gpu_id 2
python simulation.py --dimension 2 --pde kolm_flow --out_name kolm_flow --num_sims 60 --gpu_id 2
python simulation.py --dimension 2 --pde ks --test_set --out_name ks_test --num_sims 5 --gpu_id 2
python simulation.py --dimension 2 --pde gs_alpha --test_set --out_name gs_alpha_test --num_sims 3 --gpu_id 2
python simulation.py --dimension 2 --pde gs_beta --test_set --out_name gs_beta_test --num_sims 3 --gpu_id 2
python simulation.py --dimension 2 --pde gs_gamma --test_set --out_name gs_gamma_test --num_sims 3 --gpu_id 2
python simulation.py --dimension 2 --pde gs_epsilon --test_set --out_name gs_epsilon_test --num_sims 3 --gpu_id 2
python simulation.py --dimension 2 --pde decay_turb --test_set --out_name decay_turb_test --num_sims 5 --gpu_id 2
python simulation.py --dimension 2 --pde kolm_flow --test_set --out_name kolm_flow_test --num_sims 5 --gpu_id 2