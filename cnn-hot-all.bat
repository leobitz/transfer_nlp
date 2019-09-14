@ECHO OFF
for %%h in (512) do (
	for %%c in (32) do (
		for %%f in (32 64) do (
			for %%d in (-1 ) do (
				for %%l in ("arabic" "finnish" "georgian" "german" "maltese" "hungarian" "navajo" "russian" "spanish" "turkish") do (
					for %%a in (3) do ( rem task
						for %%t in (1 2 3 4 5) do (
							if exist "data\results\cnn-%%l-%%a-%%h-%%f-%%t.txt" (
								echo "file exists"
							) else (
								echo "data\results\cnn-%%l-%%a-%%h-%%f-%%t.txt"
								python cnn-onehot.py --epochs 80 --batch_size 32 --hidden_size %%h --feat_embed_size %%f  --data_size -1  > data\results\cnn-%%l-%%a-%%h-%%f-%%t.txt
							)
						)
					)
				)
			)
		)
	)
)