@ECHO OFF
for %%h in (512) do (
	for %%c in (32) do (
		for %%f in (8 16 32 ) do (
			for %%d in (14 29 44 ) do (
				for %%t in (1 2 3 4 5) do (
					if exist "data\results\%%h-%%f-%%c-%%d-%%t.txt" (
						echo "file exists"
					) else (
						echo "data\results\%%h-%%f-%%c-%%d-%%t.txt"
						python atten-onehot.py --epochs 80 --batch_size 128 --hidden_size %%h --feat_embed_size %%h --char_embed_size %%c --data_size %%d  > data\results\%%h-%%f-%%c-%%d-%%t.txt
					)
					
				)
			)
		)
	)
)