#Deep Deterministic Policy Gradient (DDPG) with Numpy


## Overview

Using from scratch

[Deep Deterministic Policy Gradient]((https://youtu.be/jgoVeAlCxJo))
## DDPG
le objects to their tracks

Kalman filter consists of two steps: prediction and update. The first step uses previous states to predict the current state. The second step uses the current measurement, such as detection bounding box location , to correct the state. The formula are provided in the following:

### Kalman Filter Equations:
#### Select action a\_t according to current policy and exploration noise
<img src="images/a_t.gif" alt="Drawing" style="width: 150px;"/>

```
a_t = actor.predict(np.reshape(s_t,(1,3)), ACTION_BOUND, target=False)+1./(1.+i+j)
```
#### Execute action a\_t and observe reward r\_t and observe new state s\_{t+1}
<img src="images/new_state.gif" alt="Drawing" style="width: 150px;"/>

```
s_t_1, r_t, done, info = env.step(a_t[0])
```
```
y=np.zeros((len(batch), action_dim))
```

```
a_tgt=actor.predict(states_t_1, ACTION_BOUND, target=True)
```

```
Q_tgt = critic.predict(states_t_1, a_tgt,target=True)
```

```
loss += critic.train(states_t, actions, y)
```
```               
a_for_dQ_da=actor.predict(states_t, ACTION_BOUND, target=False)
```
```
dQ_da = critic.evaluate_action_gradient(states_t,a_for_dQ_da)
```

```
actor.train(states_t, dQ_da, ACTION_BOUND)
```                



