### Pendulum-v0
* requires using original FPS for training.

```bash
#python main.py --debug --env=Pendulum-v0 --action_repeat 1 --noise_level 0
python main.py --debug --env=Pendulum-v0 --action_repeat 1 --noise_level 0 --batch_size 512 --cuda
```

### MountainCarContinuous-v0
```bash
python main.py --debug --env=MountainCarContinuous-v0 --action_repeat 4 --noise_level 0.1 --cuda --batch_size 512 --cuda --seed 42
```

### Pong-ram-v0
```bash
python main.py --debug --env=Pong-ram-v0 --action_repeat 4 --discrete --bn --warmup 10000
```
