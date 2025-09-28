import progressbar
import time

import snake
import exw
import compteur

# Au début de ton programme
executions = compteur.compter_executions()
print(f"Exécution n°{executions}")


agent = snake.ia.DQNAgent(snake.ia.input_dim, snake.action_dim)
score_mean = []
score_temp = 0

modulo = (snake.ia.nb_loop_train-1) // 100

fichier, wb, ws = exw.create("donnees2", "entrainement" + str(executions), "X", "Y")

model_name = "models3" # "models" + str(executions)

if snake.ia.os.path.exists(model_name + "/snake_dqn_model.pth"):
    agent.load_model(model_name + "/snake_dqn_model.pth")
    print("Model loaded : " + model_name + "/snake_dqn_model.pth")
else:
    snake.ia.os.makedirs(model_name, exist_ok=True)
    print("Model created")



for episode in progressbar.progressbar(range(snake.ia.nb_loop_train)):
    # state = [250, 353.5533905932738, 500, 424.26406871192853, 300, 353.5533905932738, 250, 353.5533905932738, 0, 0, 200, 0, 0, 0, 0, 0]

    score_temp += snake.game_loop(snake.rect_width, snake.rect_height, snake.display, agent)
    # print(f'longeur du buffer : {len(agent.replay_buffer.buffer)}')
    agent.train_step(batch_size=64)


    # while not done:
    #     action = agent.select_action(state, snake.action_dim)
    #     next_state, reward, done, _ = env.step(action)
    #
    #     # Stocke l’expérience
    #     agent.replay_buffer.push(state, action, reward, next_state, done)
    #
    #     # Entraîne le réseau
    #     agent.train_step(batch_size=64)
    #
    #     state = next_state

    # Mise à jour du réseau cible toutes les N itérations
    if episode % modulo == 0 and episode != 0:
        score_mean.append(score_temp/modulo)
        agent.update_target()

        exw.ajouter_donnee(fichier, wb, ws, episode, score_temp/modulo, "Graphe de l'évolution des scores", "Episode", "Score")

        # Sauvegarder le modèle
        agent.save_model(model_name + '/snake_dqn_model.pth')

        score_temp = 0



# print (score_mean)