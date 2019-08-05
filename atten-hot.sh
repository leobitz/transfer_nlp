export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/bin:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-10.0/bin:$PATH"
=alias tf="conda activate tf2"

# export PATH=$PATH:/home/leo/projects/amharic_word_embedding/data/fastText/build
# export PATH=$PATH:/home/leo/projects/amharic_word_embedding/word2vec

# source /home/leo/.bashrc
# cd /home/leo/projects/amharic_word_embedding/data

for hidden_size in 64 128 256 512
do
    for char_embed_size in 32 64 128 256
    do
        for feat_embed_size in 16 32 64 
        do
            for file in 14 28 56
            do
                for take in 1 2 3 4 5
                do
                    output_file="$lang-$data_size-$embed_size-$win_size-$sample"
                    input_file="$lang-$data_size"
                    if [ -e output_file ]
                    then
                        echo "File Exists"
                    else
                        # code
                    fi
                done
            done
        done
    done
done

# /home/leo/anaconda3/envs/torch/bin/python eval.py $embed_size &