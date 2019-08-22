export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/bin:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-10.0/bin:$PATH"
alias tf="conda activate tf2"

# export PATH=$PATH:/home/leo/projects/amharic_word_embedding/data/fastText/build
# export PATH=$PATH:/home/leo/projects/amharic_word_embedding/word2vec

# source /home/leo/.bashrc
# cd /home/leo/projects/amharic_word_embedding/data

for hidden_size in 256 512
do
    #for char_embed_size in 32 64 128 256
    #do
        for feat_embed_size in 8 16 32 64
        do
            for data_size in 14 29 44
            do
                for take in 1 2 3 4 5
                do
                    output_file="data/results/$hidden_size-$feat_embed_size-$data_size-$take.txt"
                    if [ -e $output_file ]
                    then
                        echo "File Exists"
                    else
                        python cnn-onehot.py --epochs 80 --batch_size 128 --hidden_size $hidden_size --feat_embed_size $feat_embed_size --data_size $data_size >  $output_file
                    fi
                done
            done
        done
    #done
done

# /home/leo/anaconda3/envs/torch/bin/python eval.py $embed_size &