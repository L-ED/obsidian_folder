
https://arxiv.org/pdf/2109.11978.pdf


On-policy RL consist of data collection and update.

Env
World cannot be reset because not every instance ended traj. So one mesh should contain all variants of env and reset carried out by changing instance place on mesh. 

Training 
Batch size must be fixed. Small BS - to much noise, Big - repetitive samples. So we need scale it in multienv $$BS = n_{robots}*n_{steps}$$where n_step is number of steps per policy update. n_robot - num of envs or policy inst.
With too big number of instances num of steps could be too low. To small steps num leads to lack of coherent temporal information which decreases learning effectiveness. Also using Generalized Advantage Estimation requires rewards from multiple time steps. Number of step is different from max traj length which is set to 20s or 1000 steps, so one ep can cover multiple updates. Optimal is between 2000 - 4000  robots and 100K- 200K batch size

To sample data from trajectory authors used mini-batch size, which splits batch size to perform backprop. This work uses ~ 10K mini-batch size for learning

Reset 
Reset on time is problem for critic learning when we calculating RTG because its unpredictable. To mitigate such problem we could add critic prediction

To train policy env increasing difficulty after each successful pass and decreasing if episode ended with less than half target velocity. After hardest level difficulty will be randomly selected. If they reached hardest level and spread across all terrains during looping? we can conclude fully learning

Observation
base vel, ang_vel, gravity vector, joint positions and velocities, previous actions and 108 surrounding terrain height measurements

Action 
Desired joint positions with PD regulator 

Reward
Encourage to follow command velocity and avoid undesired velocities along other axes. Penalize joint torques, accelerations, target changes, collisions. Add terms encouraging to take long steps.

Sim to real
Change terrain friction, add noise to measurements and random push every 10s. ANYmal uses series elastic actuator which hard to model, so authors trained actuator to compute torques from position commands. Instead of feeding some horizon of position authors used LSTM


Thoughts
Lets use online hard samples mining OHEM and increase number of robots, but it can be inefficient. 

Create env sample with different parameters(step size, slope)  (already, as seen in photo) to be used after hardest passed

![[Screenshot from 2024-03-20 18-18-57.png]]

![[Screenshot from 2024-03-20 18-19-23.png]]


![[Screenshot from 2024-03-20 18-19-34.png]]