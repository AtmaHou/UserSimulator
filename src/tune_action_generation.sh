#!/usr/bin/env bash
echo Start tuning
# === define paprameters ===
batch_size=(32 64 128)
hidden_dim=(64 128 256 384)
max_epoch=(40)
dropout=(0.2 0.5 0.8)
depth=(2 3)
learn_rate=(0.1 0.01 0.001)
learn_rate_decay=(0.9 0.99)
teacher_forcing_ratio=(0.2 0.5 0.8)

for b_s in ${batch_size[@]}
do
    for h_d in ${hidden_dim[@]}
    do
        for m_e in ${max_epoch[@]}
        do
            for d_o in ${dropout[@]}
            do
                for dep in ${depth[@]}
                do
                    for lr in ${learn_rate[@]}
                    do
                        for lr_d in ${learn_rate_decay[@]}
                        do
                            for t_f in ${teacher_forcing_ratio[@]}
                            do
                                file_mark=b_s${b_s}-h_d${h_d}-m_e${m_e}-d_o${d_o}-dep${dep}-lr${lr}-lr_d${lr_d}-t_f${t_f}
                                echo python run_action_generation.py \
                                    -sm sv2s \
                                    --batch_size ${b_s} \
                                    --hidden_dim ${h_d} \
                                    --max_epoch ${m_e} \
                                    --dropout ${d_o} \
                                    --depth ${dep} \
                                    --lr ${lr} \
                                    --lr_decay ${lr_d} \
                                    --teacher_forcing_ratio ${t_f} \
                                    --model_name ${file_mark}.model.pkl \
                                    -gpu 3
                                nohup python run_action_generation.py \
                                    -sm sv2s \
                                    --batch_size ${b_s} \
                                    --hidden_dim ${h_d} \
                                    --max_epoch ${m_e} \
                                    --dropout ${d_o} \
                                    --depth ${dep} \
                                    --lr ${lr} \
                                    --lr_decay ${lr_d} \
                                    --teacher_forcing_ratio ${t_f} \
                                    --model_name ${file_mark}.model.pkl \
                                    -gpu 3 \
                                    > ./tune/${file_mark}.log &
                            done
                        done
                        wait # Partial parallelism
                    done
                done
            done
        done
    done
done


# test wait
#echo 1
#nohup sleep 10&
#echo 3
#echo 4
#wait  #会等待wait所在bash上的所有子进程的执行结束，本例中就是sleep 5这句
#echo 5
#nohup sleep 5&
#echo 6
