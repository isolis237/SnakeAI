# manual
python main.py snake --grid-w 12 --grid-h 12 --fps 12

# only NN
python main.py policy

# NN + Snake
python main.py loop --episodes 10 --render

# 9 envs, tiled 3x3 window at 12 FPS
python main.py multi --n-envs 9 --episodes-per-env 2 --render --fps 12 --grid-w 12 --grid-h 12 --cell-px 20