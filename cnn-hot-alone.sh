export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/bin:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-10.0/bin:$PATH"
alias tf="conda activate tf2"

source /home/leo/.bashrc

# ~/anaconda3/envs/tf2/bin/python cnn-onehot.py

for hidden_size in 256 #512
do
    for kernels in 24 32 
    do
        for feat_embed_size in 16 64
        do
            for data_size in 14 44
            do
                for pool_size in 2 3
                do
                    for kernel_size in 3 5 7
                    do
                        for take in 1 2 3 4 5
                        do
                            output_file="/home/leo/projects/transfer_nlp/data/results/cnn-$hidden_size-$feat_embed_size-$data_size-$kernels-$kernel_size-$pool_size-$take.txt"
                            find  /home/leo/projects/transfer_nlp/data/results -type f -size 0 -delete
                            if [ -e $output_file ]
                            then
                                echo "File Exists"
                            else
                                echo $output_file
                                ~/anaconda3/envs/tf2/bin/python /home/leo/projects/transfer_nlp/cnn-onehot.py --epochs 80 --batch_size 128 --kernel_size $kernel_size --pool_size $pool_size --kernels $kernels --hidden_size $hidden_size --feat_embed_size $feat_embed_size --data_size $data_size >  $output_file 
                                            
                            fi
                        done
                    done
                done
            done
        done
    done
done

# /home/leo/anaconda3/envs/torch/bin/python eval.py $embed_size &