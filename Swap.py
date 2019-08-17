from Karl import Karl
from Env import Data

def Main():
	env = Data()
	karl = Karl(env)
	
	# karl.Train()
	karl.Test([100, 200, 5, 60, 70])
	
if __name__ == "__main__":
	Main()