@ECHO OFF
for %%h in (256 512) do (
	for %%c in (32) do (
		for %%f in (16 64) do (
			for %%d in (14 44 ) do (
				for %%t in (1 2 3 4 5) do (
					if exist "data\results\lstm-atten-%%h-%%f-%%c-%%d-%%t.txt" (
						echo "file exists"
					) else (
						echo "data\results\lstm-atten-%%h-%%f-%%c-%%d-%%t.txt"
						python lstm-atten-onehot.py --epochs 80 --batch_size 128 --hidden_size %%h --feat_embed_size %%h --char_embed_size %%c --data_size %%d  > data\results\lstm-atten-%%h-%%f-%%c-%%d-%%t.txt
					)
					
				)
			)
		)
	)
)