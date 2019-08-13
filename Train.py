import json
import time
from Model import Model
from Data import Data

iterations = 20000
attempts = 50

def Main():
    model = Model(iterations=iterations)
    data = Data()
    
    total_reward = 0
    last_total = 0
    last_pause = 0
    reward = 0
    correct = False
    batch_size = 16
    
    for i in range(iterations):
        state = data._state
        model.ClearMemory()
        
        for step in range(attempts):            
            action = model.NextAction(state)
            next_state, reward, correct = data.Act(action)
            model.AddMemory(state, action, next_state, reward, correct)
            state = next_state
            
        if len(model._memory) > batch_size:
            model.Replay(batch_size)
            
            
            # state = data._state
            # action = model.NextAction(state)
            # next_state, correct = data.Action(action)
            
            # if step == 9 or correct:
            #     reward = data.WaitReward(step, correct)
            #     model.Update(state, next_state, action, reward)
            
            
            # model.Update(state, next_state, action, reward)
            
            # if done:
                # print(data._state)
                # since_last = step - last_pause
                # print(since_last)
                # last_pause = step
                # data.Shuffle()
                # print(data._state)
                # time.sleep(1)
                # break
            # total_reward += reward
            # if step % 1000 == 0:
            #     pass
                # performance = (total_reward - last_total) / 250.0
                # print(json.dumps({'step': step, 'performance': performance, 'total_reward': total_reward}))
                # last_total = total_reward
        
        if correct:
            print("Iteration:", i)
            print("Reward", reward)        
            print("Completed", data._state)
            data.Shuffle()
            print("New", data._state)
            time.sleep(1.5)
    
if __name__ == "__main__":
    Main()