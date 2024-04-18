python test_frame_based.py --save_path=./pre-trained/vto --device=cuda:0 --debug=0 --mode=sequence --print_final_result=1 --n_frames=300 --autoregressive=1 --another_dataset=./data/vto-example/dress_shape00_01_01

python test_frame_based.py --save_path=./pre-trained/cloth3d --device=cuda:0 --debug=0 --mode=sequence --print_final_result=1 --n_frames=300 --autoregressive=1 --another_dataset=./data/cloth3d-example/00001_Skirt_cotton --use_heuristic_boundary=1
