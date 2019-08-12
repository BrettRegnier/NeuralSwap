import json
import time
from Model import Model
from Data import Data

iterations = 20000

def Main():
    model = Model(iterations=iterations)
    data = Data()
    
    total_reward = 0
    last_total = 0
    last_pause = 0
    
    for step in range(iterations):
        state = data._state
        action = model.NextAction(state)
        next_state, reward, done = data.Act(action)
        model.Update(state, next_state, action, reward)
        
        if done:
            print(data._state)
            since_last = step - last_pause
            print(since_last)
            last_pause = step
            data.Shuffle()
            print(data._state)
            time.sleep(1)
        total_reward += reward
        if step % 1000 == 0:
            pass
            # performance = (total_reward - last_total) / 250.0
            # print(json.dumps({'step': step, 'performance': performance, 'total_reward': total_reward}))
            # last_total = total_reward
            
            
    print(data._state)
    
if __name__ == "__main__":
    Main()