import agent
import random_agent
import visual_game_manager
import alphabeta_agent
import game_manager
import MCTSAgent
import alphabeta_agent_2


visual = False
count = 0
wins = 0
draws = 0
for i in range(20):
    if i%2==0:
        a = MCTSAgent.MCTSAgent2(1)
        b = MCTSAgent.MCTSAgent(-1)
        c = random_agent.RandomAgent(-1)
        d = alphabeta_agent.AlphaBetaAgent(-1)
        if visual:
            game = visual_game_manager.VisualGameManager(a,c)
        else:
            game = game_manager.TextGameManager(a,d,display=False)
        x,y = game.play()
        if y==1:
            wins+=1
        elif x == 0:
            draws+=1
    else:
        a = MCTSAgent.MCTSAgent2(-1)
        b = MCTSAgent.MCTSAgent(1)
        c = random_agent.RandomAgent(1)
        d = alphabeta_agent.AlphaBetaAgent(1)
        game = game_manager.TextGameManager(d,a,display=False)
        x,y = game.play()
        if x==1:
            wins+=1
        elif x == 0:
            draws+=1
    count+=1
    print(f"Ratio : {wins}/{count}")
    
    
print(f"The AI has won {wins} time(s) and there have been {draws} draw(s).")






