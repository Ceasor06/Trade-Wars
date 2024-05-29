class StockEnvironment(gymnasium.Env):
  def __init__(self, folder_path, num_agents = 2, train = True, number_of_days_to_consider = 40, split_ratio = 0.8):
    super(StockEnvironment, self).__init__()

    self.num_agents = num_agents
    self.folder_path = folder_path
    self.train = train
    self.number_of_days_to_consider = number_of_days_to_consider
    self.split_ratio = split_ratio

    self.data = self.load_data(folder_path)
    #print(f"Total Rows: {len(self.data)}")
    #print(f"Total Columns: {len(self.data.columns)}")
    #print("Columns:", self.data.columns.tolist())
    #print("Unique Symbols:", self.data['Symbol'].unique())
    #self.training_stock_data = self.stock_data.iloc[:int(0.8 * len(self.stock_data))]
    #self.testing_stock_data = self.stock_data.iloc[int(0.8 * len(self.stock_data)):].reset_index()

    split_index = int(len(self.data) * self.split_ratio)
    self.training_data = self.data.iloc[:split_index]
    self.testing_data = self.data.iloc[split_index:].reset_index(drop=True)

    self.stock_data = self.training_data if self.train else self.testing_data

    self.action_spaces = {agent_id: spaces.Discrete(3) for agent_id in range(self.num_agents)}
    self.observation_spaces = {agent_id: spaces.Discrete(4) for agent_id in range(self.num_agents)}

    self.agent_states = {agent_id: self.initialize_agent_state() for agent_id in range(self.num_agents)}

    #self.investment_capital = 10000
    #self.number_of_shares = 0
    #self.stock_value = 0
    #self.book_value = 0

    #self.total_account_value = self.investment_capital + self.stock_value
    #self.total_account_value_list = []

    if self.train:
        self.max_timesteps = len(self.training_data) - self.number_of_days_to_consider -1
    else:
        self.max_timesteps = len(self.testing_data) - self.number_of_days_to_consider -1
    self.timestep = 0
    #self.reset()

  def initialize_agent_state(self):
    return {
            "investment_capital": 10000,
            "number_of_shares": 0,
            "stock_value": 0,
            "book_value": 0,
            "total_account_value": 10000,
            "total_account_value_list": [],
        }


  def load_data(self, folder_path):
    combined_data = []
    for filename in os.listdir(folder_path):
      if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        stock_data = pd.read_csv(file_path)
        #stock_data['Symbol'] = filename.replace('_data.csv', '')
        stock_data.columns = ['Date'] + list(stock_data.columns[1:])
        symbol = filename.split('_data.csv')[0]
        stock_data['Symbol'] = symbol
        combined_data.append(stock_data)

    if not combined_data:
          raise ValueError("No data found in the folder.")

    combined_data = pd.concat(combined_data, axis=0)
    #combined_data.sort_values(by=['Symbol', 'Date'], inplace=True)
    combined_data[['open', 'high', 'low', 'close', 'volume']] = StandardScaler().fit_transform(combined_data[['open', 'high', 'low', 'close', 'volume']])
    return combined_data

  def reset(self):

    self.timestep = 0
    self.agent_states = {agent_id: self.initialize_agent_state() for agent_id in range(self.num_agents)}
    #observations = {}
    #for agent_id in range(self.num_agents):
        #observations[agent_id] = self.generate_observation(agent_id)
    observations = {agent_id: self.generate_observation(agent_id) for agent_id in range(self.num_agents)}
    print(f"Reset states: {observations}")
    return observations


  def generate_observation(self, agent_id):

    price_increase_list = []
    stock_data = self.stock_data


    #for i in range(self.number_of_days_to_consider):
     # current_price = stock_data['close'].iloc[self.timestep + i]
    #  next_price = stock_data['close'].iloc[self.timestep + 1 + i]
    #  if next_price - current_price > 0:
    #      price_increase_list.append(1)
    #  else:
    #      price_increase_list.append(0)



    for i in range(self.number_of_days_to_consider):
        current_price = stock_data['close'].iloc[self.timestep + i]
        next_price = stock_data['close'].iloc[self.timestep + 1 + i]
        if next_price - current_price > 0:
            price_increase_list.append(1)
        else:
            price_increase_list.append(0)


    price_increase = (np.sum(price_increase_list) / self.number_of_days_to_consider) >= 0.5
    stock_held = self.agent_states[agent_id]['number_of_shares'] > 0

    if price_increase and not stock_held:
        observation = 0
    elif price_increase and stock_held:
        observation = 1
    elif not price_increase and not stock_held:
        observation = 2
    else:
        observation = 3 #not price_increase and stock_held
    return observation


  def step(self,actions):
    #print(f"Current timestep: {self.timestep}")
    rewards = {}
    observations = {}
    for agent_id, action in actions.items():
      reward = self.environment_action(agent_id, action)
      rewards[agent_id] = reward
      observations[agent_id] = self.generate_observation(agent_id)
      reward = self.environment_action(agent_id, action)
      self.agent_states[agent_id]['total_account_value'] = self.agent_states[agent_id]['investment_capital'] + self.agent_states[agent_id]['stock_value']
      self.agent_states[agent_id]['total_account_value_list'].append(self.agent_states[agent_id]['total_account_value'])
      rewards[agent_id] = reward
      observations[agent_id] = self.generate_observation(agent_id)


    #print(f"Rewards calculated: {rewards}")
    self.timestep += 1
    terminated = self.timestep >= self.max_timesteps

    #for agent_id in actions.keys():
        #observations[agent_id] = self.generate_observation(agent_id)

    truncated = False
    info = {}

    return observations, rewards, terminated, truncated, info

  def render(self, mode='human'):
    """This method renders the agents' total account values over time.

    :param mode: 'human' renders to the current display or terminal and returns nothing."""
    plt.figure(figsize=(15, 10))
    for agent_id, state in self.agent_states.items():
        plt.plot(state['total_account_value_list'], label=f'Agent {agent_id}', linewidth=3)
    plt.xlabel('Days', fontsize=32)
    plt.ylabel('Total Account Value', fontsize=32)
    plt.title('Total Account Value over Time for All Agents', fontsize=38)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.show()

  def environment_action(self, agent_id, action):
    reward = 0
    agent_state = self.agent_states[agent_id]
    penalty = 0
    transaction_cost = 0
    number_of_shares_to_buy = 0

    current_price_index = self.timestep + self.number_of_days_to_consider
    if current_price_index < len(self.stock_data):
        current_price = self.stock_data['open'].iloc[current_price_index]

        #action logic
        if action == 0:  #buy shares
            if agent_state['number_of_shares'] > 0:
                penalty = -1
            number_of_shares_to_buy = math.floor(agent_state['investment_capital'] / current_price)
            transaction_cost = current_price * number_of_shares_to_buy * 0.005

            if number_of_shares_to_buy > 0:
                agent_state['number_of_shares'] += number_of_shares_to_buy
                agent_state['stock_value'] = agent_state['number_of_shares'] * current_price
                agent_state['investment_capital'] -= number_of_shares_to_buy * current_price
                agent_state['book_value'] += current_price * number_of_shares_to_buy
                reward = number_of_shares_to_buy * 0.1 - transaction_cost
            else:
                reward = -10
        elif action == 1:  #sell shares
            if agent_state['number_of_shares'] > 0:
                sell_value = current_price * agent_state['number_of_shares']
                transaction_cost = sell_value * 0.005
                profit = sell_value - agent_state['book_value'] - transaction_cost
                reward = profit
                agent_state['investment_capital'] += sell_value - transaction_cost
                agent_state['number_of_shares'] = 0
                agent_state['stock_value'] = 0
                agent_state['book_value'] = 0
            else:
               reward = -1

        elif action == 2:  #hold shares
            agent_state['stock_value'] = current_price * agent_state['number_of_shares']
            if agent_state['number_of_shares'] > 0 and agent_state['book_value'] > 0:
                unrealized_profit = agent_state['stock_value'] - agent_state['book_value']
                reward = unrealized_profit / agent_state['book_value']
            else:
                reward = -0.1

        agent_state['stock_value'] = current_price * agent_state['number_of_shares']
        agent_state['total_account_value'] = agent_state['investment_capital'] + agent_state['stock_value']
    else:
        print(f"Index {current_price_index} out of bounds for stock_data with length {len(self.stock_data)}")
        reward = 0

    return reward