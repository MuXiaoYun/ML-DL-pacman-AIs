ai_controls = 1 # 0 for not control, 1 for controling pacman, 2 for controling guard 
ai_type = 'dqn' # 'mcts' for MCTS, 'dqn' for DQN

max_moves = 100 # 游戏中允许的最大操作数

max_rollout = 20 # rollout阶段允许的最大深度
max_runtime = 15 # 单步允许最大思考时间
state_movesleft_w = 1e-3 # 鼓励智能体尽快完成任务时，对剩余步数的奖励加成权重
mcts_c = 1 # mcts的超参数c

FCHiddenSize = 128  # DQN网络的隐藏层大小

DQNsteppunish = 1e-3
DQNepsilon = 0.3  # 初始epsilon值
DQNMinEpsilon = 0.1  # 最小epsilon值
DQNEpsilonDecay = 0.997  # epsilon衰减率
DQNgamma = 0.9  # 折扣因子
DQNlr = 1e-3  # 学习率
DQNMinlr = 1e-6  # 最小学习率
DQNlrDecay = 0.9999  # 学习率衰减率
DQNbatch_size = 128  # 批处理大小
DQNepochs = 10000  # 训练的总轮数
DQNMaxBufferSize = 10000  # 经验回放缓冲区的最大大小
DQNUpdateFreq = 20  # DQN网络更新频率
DQNSaveFreq = 2000  # 保存模型的频率