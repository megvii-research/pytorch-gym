### Pendulum-v0
* requires using original FPS for training.

```bash
python main.py --debug --env=Pendulum-v0 --action_repeat 1
```

### Pong-ram-v0
```bash
python main.py --debug --env=Pong-ram-v0 --action_repeat 4 --discrete --bn --warmup 10000
```
