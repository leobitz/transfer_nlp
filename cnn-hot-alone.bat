@ECHO OFF
for %%h in (256) do (
	for %%k in (48) do (
		for %%f in (16 64) do (
			for %%d in (14 44 ) do (
				for %%p in (2 3) do (
					for %%ks in (3 5 7) do (
                       for %%t in (1 2 3 4 5) do (
                            if exist "data\results\lstm-atten-%%h-%%f-%%c-%%d-%%t.txt" (
                                echo "file exists"
                            ) else (
                                echo "data\results\lstm-atten-%%h-%%f-%%c-%%d-%%t.txt"
                                REM python lstm-atten-onehot.py --epochs 80 --batch_size 128 --hidden_size %%h --feat_embed_size %%h --char_embed_size %%c --data_size %%d  > data\results\lstm-atten-%%h-%%f-%%c-%%d-%%t.txt
                                python cnn-onehot.py --epochs 80 --batch_size 128 --kernel_size $ks --pool_size $p --kernels $k --hidden_size $h --feat_embed_size $f --data_size $d >  $output_file 
                            )
                        )
				    )
				)
			)
		)
	)
)
REM export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/bin:$LD_LIBRARY_PATH"
REM export PATH="/usr/local/cuda-10.0/bin:$PATH"
REM alias tf="conda activate tf2"

REM source /home/leo/.bashrc

REM # ~/anaconda3/envs/tf2/bin/python cnn-onehot.py

REM for hidden_size in 256 #512
REM do
REM     for kernels in 24 32 
REM     do
REM         for feat_embed_size in 16 64
REM         do
REM             for data_size in 14 44
REM             do
REM                 for pool_size in 2 3
REM                 do
REM                     for kernel_size in 3 5 7
REM                     do
REM                         for take in 1 2 3 4 5
REM                         do
REM                             output_file="data/results/cnn-$hidden_size-$feat_embed_size-$data_size-$kernels-$kernel_size-$pool_size-$take.txt"
REM                             find  ./data/results -type f -size 0 -delete
REM                             if [ -e $output_file ]
REM                             then
REM                                 echo "File Exists"
REM                             else
REM                                 echo $output_file
REM                                 ~/anaconda3/envs/tf2/bin/python cnn-onehot.py --epochs 80 --batch_size 128 --kernel_size $kernel_size --pool_size $pool_size --kernels $kernels --hidden_size $hidden_size --feat_embed_size $feat_embed_size --data_size $data_size >  $output_file 
                                            
REM                             fi
REM                         done
REM                     done
REM                 done
REM             done
REM         done
REM     done
REM done

REM # /home/leo/anaconda3/envs/torch/bin/python eval.py $embed_size &