from tucb import TUCB
import matplotlib.pyplot as plt


def main():
    
    agents = []
    nb_agents = 4
    nb_plays = 100
    actions = [0,1,2,3]

    for i in range(nb_agents) : 
        agents.append(TUCB(nb_agents - 1))

    prev_actions = []
    for t in range(nb_plays) :
        prev_actions = list(actions)
        for i in range(nb_agents) :
            #chaque agents regarde les actions des autres agents, mais pas la sienne
            actions[i]= agents[i].getNextAction(prev_actions [0:i]+prev_actions[(i+1):]) 




    #The display  
    plt.figure(figsize=(12,8))
    i=1
    for a in agents:
        plt.plot(a.cumul_regret, label = "Agent " + str(i))
        i += 1
    plt.xlabel("Plays", fontsize = 14)
    plt.ylabel("Cumulative regret", fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 14)
    plt.title("Cumulative Regret of 4 TUCB Agents in a Fully Connected Graph", fontsize = 20)
    plt.savefig("TUCB_cumul_regretpng")
    plt.show()
    

if __name__ == "__main__":
    main()