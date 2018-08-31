#!/usr/bin/env bash
echo HINT: Parallel testing. Pass int parameter to set trained agent to test.
echo e.g:
echo source eval_by_chatting.py.sh 1 2 3 4
echo or
echo source eval_by_chatting.py.sh 3 2
# === define agents to chat with ===
# agt == 0:AgentCmd  1:InformAgent 2:RequestAllAgent 3:RandomAgent 4:EchoAgent 5: RequestBasicsAgent 9: AgentDQN
#agt_lst=(1 2 3 4 5 9) # only agent 9 is available here
agt_lst=(9)

agt_model_lst[0]=./deep_dialog/checkpoints/rl_agent/agt_9_usr_1_b-e329_c-e500_s-r0.80_h-s80_sd1_epsl0.01_rftFalse.p
agt_model_lst[1]=./deep_dialog/checkpoints/rl_agent/agt_9_usr_2_b-e296_c-e500_s-r0.63_h-s80_sd1_epsl0.01.p
agt_model_lst[2]=./deep_dialog/checkpoints/rl_agent/agt_9_usr_3_b-e423_c-e500_s-r0.46_h-s80_sd100_epsl0.01_rftFalse.p
agt_model_lst[3]=./deep_dialog/checkpoints/rl_agent/agt_9_usr_4_b-e435_c-e500_s-r0.48_h-s80_sd1_epsl0.p
agt_model_lst[4]=./deep_dialog/checkpoints/rl_agent/agt_9_usr_5_b-e316_c-e500_s-r0.80_h-s80_sd1_epsl0.0.p

agt_model_id_lst=($1 $2 $3 $4 $5 $6) # eg, 0 1 2 3 4



# === define users to use ===

# 0 is a Frozen user simulator. 1 Rule based, 2 Supervised User, 3 Seq2Seq User, 4 Seq2Seq_Attention User, 5 State2Seq User
user_lst=(1 2 3 4 5)  # load from parameters, and number params could be accepted

# === Loop for all case and run ===
for agt in ${agt_lst[@]}
do
	for agt_model_id in ${agt_model_id_lst[@]}
	do
		for user in ${user_lst[@]}
        do
    		export OMP_NUM_THREADS=5  # threads num for each task
#    		echo nohup python run.py --agt ${agt} --usr ${user} --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 500 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 2 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --trained_model_path ${agt_model_lst[agt_model_id]} -rft > ./cross_eval/agt${agt}_by_user${agt_model_id}_user${user}.log &
#    		nohup python run.py --agt ${agt} --usr ${user} --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 500 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 2 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --trained_model_path ${agt_model_lst[agt_model_id]} -rft > ./cross_eval/agt${agt}_by_user${agt_model_id}_user${user}.log &

    		echo nohup python run.py --agt ${agt} --usr ${user} --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 500 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 2 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --trained_model_path ${agt_model_lst[agt_model_id]} > ./cross_eval/agt${agt}_by_user${agt_model_id}_user${user}.log &
    		nohup python run.py --agt ${agt} --usr ${user} --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 500 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 2 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --trained_model_path ${agt_model_lst[agt_model_id]} > ./cross_eval/agt${agt}_by_user${agt_model_id}_user${user}.log &
#            echo ./cross_eval/agt${agt}_by_user${agt_model_id}_user${user}.log 233333333 ${agt_model_lst[agt_model_id]}> ./cross_eval/agt${agt}_by_user${agt_model_id}_user${user}.log
        done
	done
done
