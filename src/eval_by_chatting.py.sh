

# === define agents to chat with ===
# agt == 0:AgentCmd  1:InformAgent 2:RequestAllAgent 3:RandomAgent 4:EchoAgent 5: RequestBasicsAgent 9: AgentDQN
agt_lst=(1 2 3 4 5)

# === define users to use ===
# 0 is a Frozen user simulator. 1 Rule based, 2 Supervised User, 3 Seq2Seq User, 4 Seq2Seq_Attention User
user_lst=(1 2 3 4)

# === Loop for all case and run ===
for agt in agt_lst
do
	for user in user_lst:
	do
		cmd = "nohup python run.py --agt ${agt} --usr ${user} --max_tur n 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 --experience_replay_pool_size 1000 --episodes 500 --simulation_epoch_size 100 --write_model_dir ./deep_dialog/checkpoints/rl_agent/ --run_mode 2 --act_level 0 --slot_err_prob 0.00 --intent_err_prob 0.00 --batch_size 16 --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p --warm_start 1 --warm_start_epochs 120 > agt${agt}_user${user}.log &"
		&{cmd}
	done
done
