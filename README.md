### Rectified Flow

```bash
python run.py flow_main
```

<div style="display: flex; justify-content: space-between;">
    <img src="static/flow_eval.png" alt="flow_eval" style="width: 42%; height: auto;">
    <img src="static/eval_frames.gif" alt="flow_eval" style="width: 42%; height: auto;">
</div>

<div style="display: flex; justify-content: space-between;">
    <img src="static/flow_sample.png" alt="flow_sample" style="width: 42%; height: auto;">
    <img src="static/sample_frames.gif" alt="flow_sample" style="width: 42%; height: auto;">
</div>

<div style="display: flex; justify-content: center;">
    <img src="static/loss.png" alt="loss" style="width: 48%;">
</div>

---

### Note:

- The loss is set to `mse - var`. This is to get a sense of how good the model is trained by looking at the loss (loss = 0 means fully trained). Loss sometimes goes negative as a result of stochastic nature of the mean and variance estimates.

- You might want to increase `config.plot_every` to get faster training.
