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

### BipedalWalker-v2
```bash
main.py --debug --env=BipedalWalker-v2 --action_repeat 1 --noise_level 1 --epsilon 1000 --batch_size 512 --crate 1e-5 --prate 1e-5 --cuda --seed 42 --discount 0.975
```

### Pong-ram-v0
```bash
python main.py --debug --env=Pong-ram-v0 --action_repeat 4 --discrete --bn --warmup 10000
```
