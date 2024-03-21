https://arxiv.org/pdf/2212.03238.pdf

Instead of env and reward tuning, lets create a policy that encodes family of locomotion strategies Multiplicity of Behavior(MOB) and choose between them in realtime w/o tuning.

Sim to real transfer can be accomplished by terrain properties estimation from sensors. To learn such policy we need randomize env parameters during training. Problems comes with unmodeled parameters. Assumption in MOB that some of policies will succeed in the out-of-distribution target env. To demonstrate this authors learned policy on flat ground and evaluated on non-flat terrain and new task. Operator can tune behavior  

Aux Reward
Simple velocity reward performs bat in sim-to-real transfer. It's necessary to use auxiliary reward that bias the robot  to maintain desired behavior(contact schedule, action smoothness, energy consumption, foot clearance) to compensate sim-to-real gap. Actually strictly specify robot behavior, authors says it's generalization because it can walk in real. Problem with in task-specific reward tuning and difficulty in designing reward for generalization to multiple behaviors. 

Parameterized locomotion 
Include gait parameters though parameter-augmented aux reward terms: tracking reward for contact patterns and target foot params. One work control timing offset between feet and swing freq. Another approach to use imitation learning from reference. 

Diversity objective
Quality diversity methods learn diff beh by enforcing a novelty objective defined among strategies and perform optimization by evolutionary strategy. Another approach ise unsupervised objective for skill discovery to improve optimization or out of distr gener. Unsup have not scaled to real world.

Hierarchical control with gait parameters
Some works learn high-level policy for gait parameters modulation of low-level model-based controller. WTW revisit hier approach to learn low and high levels

Method
Learn conditional policy $$\pi(a|c_t, b_t)$$
 where c - command, b - behavior parameters, to track velocities in **body-frame** axes $$c=[{v_x}^{cmd}, {v_y}^{cmd}, {w_z}^{cmd}]$$
This work use human-readable parameters corresponding to gait  properties $$b=[\theta_1^{cmd}, \theta_2^{cmd}, \theta_3^{cmd}, f^{cmd}, h_z^{cmd}, \phi^{cmd}, s_y^{cmd}, h_z^{f,cmd}]$$
$$\theta^{cmd} -timing \quad offsets \quad between \quad feet \quad pairs$$ f  - stepping frequency in Hz. h - body height command, phi - body pitch, s - robot stance width, h^t - footswing height.

Reward
Rew = task reward + Augmented aux + Fixed Aux
Task reward is velocity tracking reward. For avoiding early termination or task abandoning due to penalties authors enforce total reward be positive linear function of task reward with aux reward scaling so agent  always rewarding for progress 
$$r_task*exp(c*r_{aux})$$
Aug aux reward are functions of beh vector. Reward increase when beh matches b and not conflict with task rew. Example of conflicting is stance width? when we counting reward as distance between legs, which will penalize robot on fast turnings. So authors implemented Raibert Heuristic, which suggest kinematic motion of the feet by computing desired foot position $$p_{x, y, foot}^{f, cmd}(s_y^{cmd})$$as an adjustment to the baseline stance width. To define desired contact schedule $$C_{foot}^{cmd}(\theta^{cmd}, t)$$  computes contact state of each foot from the phase and timing variable

Training
Resample desired task within each training episode using **adaptive curriculum strategy**
timing offsets are sampled as one of the symmetric quadrupedal contact patterns(pronking, trotting, bounding, pacing). Other params sampled uniformly. Vary body inertia, motor strength, joint positions calibration, ground friction and gravity vector. No need for motor latency varying if it constant. So they used actuator network to capture PD error and torque. Separately identified latency 20ms and modelled this.

Trained to minimize energy by joint speed and torque max(torque*speed, 0)

Policy input
30 - step history of observations, commands, behaviors, previous actions, timing variables. observation - joint positions and velocities and gravity vector in the body frame. Timing reference variables $$t =[sin(2\pi*t^{FR}), sin(2\pi*t^{FL}), sin(2\pi*t^{RR}), sin(2\pi*t^{RL})]$$
computed from offset timings of each foot $$[t^{FR}, t^{FL}, t^{RR}, t^{RL}]=[t+\theta_2^{cmd}+\theta_3^{cmd}, t+\theta_1^{cmd}+\theta_3^{cmd}, t+\theta_1^{cmd}, t+\theta_2^{cmd}]$$
where t - counter variable advances from 0 to 1 during gate cycle  and FR, FL, RR, RL - legs

Policy architecture
MLP [512, 256, 128] with ELU activations. Policy input also includes estimated domain parameters: ground friction and robot body velocity, which predicted from observation history using supervised learning by MLP[256,128] ELU estimator

Action Space
position targets for each twelve joints, which tracke using PD controller with kp=20, kd=0.5
