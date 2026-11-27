Open normal terminal
scp "group2@rockhill.csis.ul.ie:~/jj_proj/JJ_Proj/videos/training/*.mp4" .



  Updated: 23 features, 191 fixes, 102 other changes
  code.claude.com/docs/en/changelog for details


❯ /clear

❯ explain why training stalls after 10 epochs at 140 reward

  Read 1 file, ran 2 shell commands

Training isn't stalling at 140. The agent is still playing at random level, and the script stops before real learning could start. A random agent scores about 140–150 per episode in Space Invaders, the same baseline your main.py measures. So the curve flattening at 140 means the network isn't yet adding anything over random play. I worked this out from rl_project.py; I haven't seen your training logs.

Why it sits at random level

1. The training budget is far too small for DQN.
EPOCHS = 100 episodes is roughly 50–80k environment steps. With replay() called every 4 steps after step 5000, that's only about 12–20k gradient updates. The original DQN paper trained Space Invaders for about 12.5M steps, and at 100k steps DQN agents are typically still close to random on this game. With this budget, a flat curve is normal.

2. By episode 10 the agent has barely trained.
- Learning only starts after step > 5000 (rl_project.py:226), which is around episode 6–8.
- Epsilon decays once per step: 0.99 × 0.9999^step. At about 7k steps it is still around 0.5, so half the actions are random. It doesn't reach EPSILON_MIN = 0.05 until about 29,850 steps.
- So by episode 10 you have a few hundred updates and a mostly random policy.

3. The plotted curve hides change.
scores is a running mean over up to 100 episodes (deque(maxlen=100)). Every point after episode 10 still includes those early random episodes, so a real improvement in individual episodes would barely move the line. Print or plot the per-episode score i alongside the mean to see what's actually happening.

4. The reward signal is weak.
- np.sign(reward) makes every alien worth +1. That's standard DQN practice, but it means the reward is just "shoot something."
- Losing a life gives no penalty and doesn't end the transition, so the agent gets no signal to dodge.
- The usual early local optimum is "sit and fire," which scores about the same as random.

A practical problem that will hurt longer runs

Each transition stores state and next_state as float32 (1, 84, 84, 4) tensors, about 113 KB each. With deque(maxlen=20_000), the full buffer is about 4.5 GB of RAM. That could cause swapping on the lab machines, and it's the main obstacle to training longer.

What would actually move the score

1. Train much longer. Aim for at least 1–2M steps rather than 100 episodes. To make that affordable:
   - Store frames as uint8 (0–255) and divide by 255 only at sample time. That cuts memory by 4×.
   - Store each frame once instead of whole stacks in both state and next_state.
   - Replace model.predict + model.fit in replay() with a single @tf.function train step. That's much faster per update.
2. Decay epsilon over a fixed horizon, for example linearly from 1.0 to 0.1 over the first 250k–1M steps, and don't start decaying until learning begins.
3. Grow the replay buffer to 100k+ (feasible once frames are uint8). Use a learning_starts of about 20–50k, and update the target network every 5–10k steps instead of every 1000.
4. Treat losing a life as terminal in the stored transition, using the info["lives"] counter, to give the agent a reason to dodge.
5. Log per-episode scores and greedy evaluation scores, not only the 100-episode running mean.

Change 1 matters most; the others make long training effective. Want me to make these changes to rl_project.py?

Cooked for 45s · done 4:29 PM

❯ so what changes would you propose

The plan is to train for about 1M steps instead of 100 episodes. Most of the changes below are there to make that affordable on the lab machines: using less memory, running each update faster, and saving progress. Nothing is applied yet, and the code isn't tested.

1. Store frames as uint8 in a preallocated buffer

Return uint8 from preprocess_state and keep each frame stack as a plain numpy array instead of a float32 tensor:

def preprocess_state(img):
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, (84, 84))
    return tf.cast(img, tf.uint8)[..., 0].numpy()          # (84, 84) uint8

def combine_images(stack, frame, m=4):
    if stack is None:                                       # first frame of an episode
        return np.stack([frame] * m, axis=-1)               # (84, 84, 4)
    return np.concatenate([stack[..., 1:], frame[..., None]], axis=-1)

Replace the deque with a ring buffer. It stores only the newest frame for next_state, since the rest of next_state is just state shifted by one frame:

class ReplayBuffer:
    def __init__(self, capacity):
        self.states      = np.zeros((capacity, 84, 84, 4), np.uint8)
        self.next_frames = np.zeros((capacity, 84, 84), np.uint8)
        self.actions     = np.zeros(capacity, np.int32)
        self.rewards     = np.zeros(capacity, np.float32)
        self.dones       = np.zeros(capacity, np.float32)
        self.capacity, self.idx, self.size = capacity, 0, 0

    def add(self, s, a, r, next_frame, done):
        i = self.idx
        self.states[i], self.next_frames[i] = s, next_frame
        self.actions[i], self.rewards[i], self.dones[i] = a, r, done
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        j = np.random.randint(0, self.size, n)
        s = self.states[j]
        ns = np.concatenate([s[..., 1:], self.next_frames[j][..., None]], axis=-1)
        return s, self.actions[j], self.rewards[j], ns, self.dones[j]

Memory use is about 35 KB per transition instead of about 225 KB now. A 100k buffer needs about 3.5 GB and a 50k buffer about 1.8 GB. Pick based on the lab machines' RAM.

2. Scale pixels inside the model

Add a rescaling layer at the start of build_model, so the buffer can stay uint8:

layers.Input(shape=(height, width, channels)),
layers.Rescaling(1.0 / 255),
layers.Conv2D(32, 8, strides=4, activation="relu"),
...

Using relu instead of elu is optional; it's what the reference DQN uses.

3. Replace predict + fit with one compiled train step

Right now each update makes three separate Keras calls. A single tf.function is much faster. It also adds Double DQN, which is a two-line change that reduces Q-value overestimation:

self.optimizer = optimizers.Adam(learning_rate=1e-4, clipnorm=10.0)
self.loss_fn = tf.keras.losses.Huber()

@tf.function
def train_step(self, s, a, r, ns, d):
    s, ns = tf.cast(s, tf.float32), tf.cast(ns, tf.float32)
    best = tf.argmax(self.model(ns), axis=1)                       # online net picks the action
    q_next = tf.gather(self.model_target(ns), best, batch_dims=1)  # target net scores it
    target = r + self.gamma * (1.0 - d) * q_next
    with tf.GradientTape() as tape:
        q = self.model(s, training=True)
        q_a = tf.reduce_sum(q * tf.one_hot(a, self.action_size), axis=1)
        loss = self.loss_fn(target, q_a)
    grads = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    return loss

choose_action becomes self.model(tf.cast(state[None], tf.float32))[0].numpy().argmax().

4. Decay epsilon linearly, starting when learning starts

def epsilon_at(step):
    frac = min(1.0, max(0, step - LEARNING_STARTS) / EPS_DECAY_STEPS)
    return EPS_START + frac * (EPS_END - EPS_START)

5. Treat a lost life as terminal in the stored transition

The game itself keeps running; only the stored done flag changes:

lives = info["lives"]                       # after env.reset()
...
obs, reward, terminated, truncated, info = env.step(action)
life_lost = info["lives"] < lives
lives = info["lives"]
buffer.add(state, action, np.sign(reward), frame, float(terminated or life_lost))
done = terminated or truncated              # this still ends the episode loop

This also fixes a small existing issue: a truncated episode will no longer be stored as terminal.

6. Count training in steps, log per episode, save checkpoints

TOTAL_STEPS      = 1_000_000
BUFFER_SIZE      = 100_000     # or 50_000 if RAM is tight
LEARNING_STARTS  = 50_000
TRAIN_FREQ       = 4
TARGET_UPDATE    = 10_000
BATCH_SIZE       = 32
GAMMA            = 0.99
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 250_000

The loop becomes while step < TOTAL_STEPS:, with episodes running inside it. Every 10 episodes, print:
- the last episode's score
- the mean of the last 20 episodes
- the current epsilon
- the step count

Every 100k steps, call self.model.save_weights(f"checkpoints/dqn_{step}.weights.h5"), so a lab session that dies doesn't lose the whole run. Change the video trigger to ep % 100 == 0, or you'll get thousands of videos.

What to expect

On these settings, scores usually only move clearly above 140 after roughly 200–500k steps, once epsilon is low and the buffer is full. A good sign is a mean around 300–600 by 1M steps. If the curve is still flat at about 150 after 300k steps, look for a bug before tuning anything.

Want me to apply these changes to rl_project.py on the lab_machines branch? I'd do a short smoke run of about 2k steps first to confirm it runs end to end.

Sautéed for 47s · done 4:31 PM

❯ so does my stuff suck rn

No. The structure is right, but the run is far too short and the replay memory is too heavy to make it longer. The design isn't the problem.

What's already right:
- Preprocessing: greyscale, 84×84, a stack of 4 frames.
- Network: the same three conv layers as the original DQN paper.
- Replay buffer and target network: both present, and the target network is synced on a schedule.
- Reward clipping and Huber loss: both standard DQN choices.
- Bellman target: reward + gamma * max(q_next), with no bootstrapping on terminal states, is correct.
- Evaluation: a greedy evaluation run with video recording, which a lot of student projects skip.

Many first DQN attempts have a real bug, like a wrong target, broken frame stacking or no target network. I didn't find one like that in yours.

What's holding it back:
- Too little training. 100 episodes is roughly 0.5% of the usual DQN budget, so a score near random is the expected result right now, not a sign it's broken.
- Too much memory. Float32 frames make the replay buffer about 4.5 GB, which makes longer runs impractical.
- Smaller tuning issues. Epsilon decays too fast and starts too early, and the running mean hides any change.

The one real bug is the slight done/truncated mix-up, and it rarely triggers. Since your code works but has only had about 1/200th of the training it needs, a flat 140 doesn't tell you anything yet about whether the approach works.

For the report, that's a fair way to present the current result: the implementation matches standard DQN, and the plateau comes from the training budget and memory limits. Then show what happens when you scale it up.

Crunched for 8s · done 4:34 PM

❯ ok but storage and compute isnt a limit, so is it all ok

Not quite. If compute and storage really aren't limits, you can skip the memory and speed work: the float32 buffer and the predict/fit calls are only about cost. But just raising EPOCHS still won't be enough, because a few settings would cap the score however long it runs.

Changes that still matter:

1. Epsilon decays far too fast for a long run. At 0.9999 per step it reaches the 0.05 minimum after about 30k steps, so by about episode 40 the agent has almost stopped exploring. A long run would spend nearly all of its time exploiting a policy it learned from very little data. Decay over about 250k steps instead.
2. The replay buffer is too small. maxlen=20_000 is only a few dozen episodes, so the agent keeps relearning recent experience. Raise it to 100k–1M. With your float32 frames that's about 22 GB of RAM at 100k, so check the machine actually has that.
3. The target network updates too often. Syncing every 1000 steps makes the targets move a lot, which tends to make Q-values unstable. Use 10k.
4. Learning starts too early. At 5000 steps the buffer barely has any data. Start at about 50k.
5. Losing a life costs nothing. The agent has no reason to dodge, so it tends to plateau at "sit and fire." Store a lost life as terminal (info["lives"]).
6. Train by steps, not episodes. As the agent improves, episodes get longer, so a fixed episode count is a poor measure of training. Aim for 1M+ steps.

Changes for the report:

- Per-episode scores. Log and plot them next to the mean; the 100-episode running mean hides change.
- Fewer videos. Record every 100 episodes, not every 5, or you'll get thousands of videos.
- Checkpoints. Save weights periodically so a crash on a lab machine doesn't lose the run.

Items 1–5 are mostly changes to constants plus about five lines for item 5, and together with item 6 they're what separates "runs longer" from "actually learns."

One caveat on "compute isn't a limit": wall-clock time still is. The current predict + fit update is slow, so 1M steps could take most of a day. The compiled train step from earlier would make it several times faster, but you can skip it if a day is fine.

Want me to make those hyperparameter and life-loss changes now?

Sautéed for 14s · done 4:34 PM

recap: You're working out why your Space Invaders DQN plateaus at about 140, which is random-level play, and the fix is longer training plus some hyperparameter and life-loss changes. Next step is your call: should I apply those changes to rl_project.py? (turn off recaps in /config)

❯ ok what about now

  Ran 2 shell commands

Much better. Most of the training settings are now in good shape, but the new 500k buffer will run the machine out of memory long before training finishes.

What's now fixed:
- Epsilon now decays at 0.99999 per step and reaches 0.05 at about 300k steps. That's a sensible exploration period.
- Learning starts at 50k steps and the target network syncs every 10k. Both are standard values.
- EPOCHS = 1000 is roughly 0.7–1M steps. That's in the range where DQN should clearly beat random.
- Logging each episode's reward and epsilon means you'll actually see progress.

The blocking issue: memory

Each stored transition is still about 225 KB of float32, because both state and next_state are (1, 84, 84, 4). With maxlen=500_000:

┌────────────────────────────┬─────────┐
│      Buffer contents       │ Memory  │
├────────────────────────────┼─────────┤
│ 50k (when learning starts) │ ~11 GB  │
├────────────────────────────┼─────────┤
│ 100k                       │ ~22 GB  │
├────────────────────────────┼─────────┤
│ 500k (full)                │ ~113 GB │
└────────────────────────────┴─────────┘

There's a second problem that is probably worse. I believe those states are stored as TensorFlow tensors that live on the GPU, not in system RAM. combine_images ends with tf.expand_dims, and with a GPU visible TensorFlow will place that result on the GPU by default. With set_memory_growth on, the replay buffer would then gradually fill GPU memory. An 8–12 GB card could run out before step 50k, which means before training even begins. I haven't run it to confirm.

Smallest fix, about 4 lines: keep frames as numpy on the CPU and store them as uint8.

# combine_images: return numpy instead of a TF tensor
return np.expand_dims(im, 0)                                  # was tf.expand_dims

# remember: compress to uint8
def remember(self, state, action, reward, next_state, done):
    self.memory.append(((state * 255).astype(np.uint8), action, reward,
                        (next_state * 255).astype(np.uint8), done))

# replay: convert back
states = np.concatenate([s for s, _, _, _, _ in minibatch]).astype(np.float32) / 255
next_states = np.concatenate([n for _, _, _, n, _ in minibatch]).astype(np.float32) / 255

state comes back as a TF tensor from preprocess_state, so you may need np.asarray(state) inside remember. With this change a full 500k buffer is about 28 GB of RAM. If the lab machines have less than that, use maxlen=200_000, which is about 11 GB.

Still worth doing

1. Store a lost life as terminal (info["lives"]), so the agent learns to dodge instead of plateauing at "sit and fire."
2. Store terminated, not done, in remember. Right now a time-limit truncation is treated as a real game over.
3. Save checkpoints, for example self.model.save_weights(...) every 50 episodes. A run that takes several hours shouldn't be lost to one crash.

The memory fix is required for the run to finish at all. Items 1–3 then improve how well the agent learns. Want me to make the memory fix and those three on the branch?

Cooked for 35s · done 4:43 PM

❯ i have a H100

An H100 has 80 GB of memory, so it delays the problem rather than removing it. A full 500k buffer at about 225 KB per transition needs about 113 GB.

┌───────────────────────┬─────────┬────────────────────────────────┐
│    Buffer contents    │ Memory  │          H100 (80 GB)          │
├───────────────────────┼─────────┼────────────────────────────────┤
│ 50k (learning starts) │ ~11 GB  │ fine                           │
├───────────────────────┼─────────┼────────────────────────────────┤
│ 250k                  │ ~56 GB  │ fine                           │
├───────────────────────┼─────────┼────────────────────────────────┤
│ ~330k                 │ ~75 GB  │ runs out of memory around here │
├───────────────────────┼─────────┼────────────────────────────────┤
│ 500k                  │ ~113 GB │ doesn't fit                    │
└───────────────────────┴─────────┴────────────────────────────────┘

So training would start fine and look healthy for hours, then crash about a third of the way through. That's probably around episode 400, depending on episode length.

Check where the buffer actually lives first, since I haven't confirmed it. Add one line in train() after building the first state:

print(state.device)   # ".../GPU:0" means the buffer is in H100 memory, ".../CPU:0" means system RAM

If it prints CPU, the same numbers apply to the server's system RAM instead.

Options:
- No code change: set maxlen=250_000, about 56 GB, which fits on the H100 with room to spare. That's still a large buffer for DQN, so the cost to learning is small.
- Keep 500k: make the uint8 + numpy fix from my last message. A full buffer is then about 28 GB in system RAM, and the GPU only holds the network.

The H100 won't make this much faster. Your bottleneck is per-step overhead on the CPU side, not GPU compute:
- environment stepping
- eager TF preprocessing on every frame
- two predict calls plus a fit call for each update, on a batch of only 32

A small CNN on batch 32 barely uses an H100, so expect roughly the same wall-clock time you'd get on a much smaller GPU. The compiled train_step I suggested earlier is the change that would actually speed it up, if that matters to you.

If you just want the run to finish, maxlen=250_000 is the one-line fix.

Cooked for 12s · done 4:45 PM
                                                                                                                       ✔ Update installed · Restart to update
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
❯ set maxlen to 250k
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ auto mode on (shift+tab to cycl




 how can i add improved logging to write information to file and maybe even pickel dump the model

I'd use three outputs:
- a text log for everything you print now
- a CSV with one row per episode, for your report plots
- saved Keras model files for the network itself

Don't pickle the model. Keras models don't pickle reliably, and model.save("x.keras") is the supported way to save one, with a one-line load. Use pickle only for plain Python data like the score history.

Here's the code for each piece. I haven't run it.

1. A folder per run (put this at the top, after the imports)

import os, csv, json, pickle, logging
from datetime import datetime

RUN_DIR = f"runs/{datetime.now():%Y%m%d_%H%M%S}"
os.makedirs(f"{RUN_DIR}/checkpoints", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(f"{RUN_DIR}/train.log"),   # to file
              logging.StreamHandler()],                       # and still to the terminal
)
log = logging.getLogger("dqn")

Each run gets its own folder, for example runs/20261124_171500/, so runs never overwrite each other. Replace your print(...) calls with log.info(...).

2. Return the loss from replay()

history = self.model.fit(states, q, verbose=0)
return history.history["loss"][0]

Loss is worth logging. If it climbs steadily or turns into nan, training has gone unstable.

3. Log each episode in train()

def train(self):
    scores = deque(maxlen=100)
    avg_scores = []
    step = 0
    best_mean = -np.inf

    csv_file = open(f"{RUN_DIR}/episodes.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["episode", "total_steps", "episode_steps", "reward",
                     "mean_100", "epsilon", "mean_loss"])

    for e in range(self.epochs):
        ...
        losses, ep_steps = [], 0
        while not done:
            ...
            ep_steps += 1
            if step > 50000 and step % 4 == 0:
                loss = self.replay(32, self.memory)
                if loss is not None:
                    losses.append(loss)
            ...

        scores.append(i)
        mean_score = np.mean(scores)
        avg_scores.append(mean_score)
        mean_loss = np.mean(losses) if losses else ""

        writer.writerow([e, step, ep_steps, i, mean_score, self.epsilon, mean_loss])
        csv_file.flush()                       # so the file is up to date if the run crashes
        log.info(f"ep {e}  steps {step}  reward {i:.0f}  mean100 {mean_score:.1f}  "
                 f"eps {self.epsilon:.3f}  loss {mean_loss}")

        # checkpoints
        if e % 50 == 0:
            self.model.save(f"{RUN_DIR}/checkpoints/ep{e}.keras")
        if mean_score > best_mean and step > 50000:
            best_mean = mean_score
            self.model.save(f"{RUN_DIR}/best.keras")

    csv_file.close()
    self.model.save(f"{RUN_DIR}/final.keras")
    with open(f"{RUN_DIR}/history.pkl", "wb") as f:
        pickle.dump({"avg_scores": avg_scores}, f)
    return avg_scores

The step > 50000 check on best.keras stops the early random episodes from counting as "best."

4. Save the hyperparameters (just before dqn.train())

with open(f"{RUN_DIR}/config.json", "w") as f:
    json.dump({"epochs": EPOCHS, "epsilon": EPSILON, "epsilon_decay": EPSILON_DECAY,
               "epsilon_min": EPSILON_MIN, "lr": LEARNING_RATE, "gamma": GAMMA,
               "buffer": dqn.memory.maxlen}, f, indent=2)

That way every result folder records the settings that produced it, which helps when you compare runs in the report.

5. Put videos and plots in the run folder too

video_folder=f"{RUN_DIR}/videos/training"   # in train_env
video_folder=f"{RUN_DIR}/videos/eval"       # in eval_env
plt.savefig(f"{RUN_DIR}/avg_scores.png", dpi=150)

Also log the eval scores with log.info(f"Eval episode {ep}: score {total}").

Loading a saved model later

model = tf.keras.models.load_model("runs/20261124_171500/best.keras")
q_values = model(state, training=False)

You can use this to make new eval videos without retraining.

What you'll get

runs/20261124_171500/
├── config.json
├── train.log
├── episodes.csv        ← plot this for the report (pandas.read_csv)
├── history.pkl
├── best.keras
├── final.keras
├── avg_scores.png
├── checkpoints/ep0.keras, ep50.keras, ...
└── videos/training/, videos/eval/

To copy a whole run back from the lab machine:

scp -r "group2@rockhill.csis.ul.ie:~/jj_proj/JJ_Proj/runs/<run_name>" .

Each .keras file is only a few MB, so checkpointing every 50 episodes is cheap. Don't save the replay buffer; it's tens of GB.

Want me to apply this to rl_project.py?

Crunched for 23s · done 9:12 PM

recap: You're getting your Space Invaders DQN to train properly on the lab H100, and we just drafted file logging and model saving for `rl_project.py`. Next, tell me whether to apply that logging plus the replay-buffer memory fix.

❯             pickle.dump({"avg_scores": avg_scores}, f)
   what does this do

It saves the list of average scores to a file, so you can load it back into Python later without retraining.

- {"avg_scores": avg_scores} is a dictionary holding one item: your list of running-mean scores, one number per episode. Wrapping it in a dictionary lets you add more items later without changing how you load it.
- pickle.dump(..., f) converts that dictionary into bytes and writes them to f, which was opened as history.pkl in binary write mode ("wb").

To get it back later, for example in a notebook when making plots:

import pickle

with open("runs/20261124_171500/history.pkl", "rb") as f:
    history = pickle.load(f)

avg_scores = history["avg_scores"]    # the same list you had at the end of training
plt.plot(avg_scores)

You get the exact Python object back, so there's no parsing to do.

You probably don't need it. The mean_100 column in episodes.csv already contains the same numbers, along with each episode's reward, epsilon and loss. And the CSV can be opened in Excel, read with pandas.read_csv, and inspected while training is still running.

The pickle is only useful if you later want to save Python objects that don't fit in a table, like nested dicts or numpy arrays. Otherwise you can delete those two lines and use the CSV.

One general caution: only load pickle files you created yourself. Loading a pickle can run arbitrary code, so never open one from an untrusted source.

Cooked for 7s · done 9:20 PM

❯ where is our model saved

  Searched for 2 patterns, read 1 file

You've added the saving code, so the model is saved inside the run folder. Each run gets its own folder, for example runs/20261124_171500/, named after the time the script starts. On the lab machine that's under ~/jj_proj/JJ_Proj/runs/.

┌──────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────┐
│                 File                 │                                When it's written                                 │                           What it is                            │
├──────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
│ checkpoints/ep0.keras, ep50.keras, … │ Every 50 episodes (rl_project.py:277)                                            │ A snapshot of the model at that point                           │
├──────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
│ best.keras                           │ Whenever the 100-episode mean hits a new high after step 50k (rl_project.py:279) │ Overwritten each time, so it always holds the best model so far │
├──────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
│ final.keras                          │ Once, after the last episode (rl_project.py:290)                                 │ The model at the end of training                                │
└──────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────┘

Use best.keras for evaluation and videos, since the end-of-training model isn't always the best one. To load it:

model = tf.keras.models.load_model("runs/<run_name>/best.keras")

Two things to check in your version:
- The "solved" early return skips the final save. The return avg_scores at rl_project.py:285 exits before final.keras, history.pkl and csv_file.close(). It only triggers if the mean reaches 9000, which won't happen, but deleting that return (or saving before it) makes the function safe either way.
- The memory issue is still there. combine_images still ends with tf.expand_dims (rl_project.py:134), so the 500k buffer will run out of memory partway through. If that happens before the end, you'll only have the checkpoints and best.keras saved up to that point, with no final.keras and no eval videos. Change it to np.expand_dims and use the uint8 storage or maxlen=250_000 from earlier.

Crunched for 17s · done 9:21 PM
                                                                                                                                                                                  ✔ Update installed · Restart to update
──────────────────────────────────────
