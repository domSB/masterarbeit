import random
import numpy as np

artikel = int(str(random.sample(set(simulation.possibles), 1)[0])[-6:])
absaetze = simulation.statistics.absaetze(artikel)
actions = simulation.statistics.actions(artikel)
fehlmenge = simulation.statistics.fehlmenge(artikel)
abschriften = simulation.statistics.abschrift(artikel)

agent_states = []
regression_states = []
full_state, info = simulation.reset()
predict_state = predictor.predict(full_state['RegressionState'])
regression_states.append(full_state['RegressionState'])
agent_state = {
    'predicted_sales': predict_state,
    'current_stock': full_state['AgentState'],
    'article_info': full_state['RegressionState']['static_input'].reshape(-1)
}
agent_states.append(agent_state)
days = 0
while True:
    # Train
    print('Day:', days)
    action = agent.act(agent_state)
    reward, fertig, new_full_state = simulation.make_action(action)
    new_predict_state = predictor.predict(new_full_state['RegressionState'])
    regression_states.append(full_state['RegressionState'])
    new_agent_state = {
        'predicted_sales': new_predict_state,
        'current_stock': new_full_state['AgentState'],
        'article_info': new_full_state['RegressionState']['static_input'].reshape(-1)
    }
    agent.remember(agent_state, action, reward, new_agent_state, fertig)
    agent_states.append(agent_state)
    agent_state = new_agent_state
    days += 1
    if fertig:
        break

for pred, true in zip(agent_states[:20], simulation.kristall_glas[:20]):
    print('True', np.argmax(true, axis=1))
    print('Pred', np.argmax(pred['predicted_sales'], axis=1), '\n')


# Spiele selber
rewards = []
while True:
    aktion = input('Aktion')
    r, done, state = simulation.make_action(int(aktion))
    print('Belohnung', r, '\nStatus', state['AgentState'])
    if done:
        print('Fertig')
        break
    rewards.append(r)
    prediction = predictor.predict(state['RegressionState'])
    print(np.argmax(prediction, axis=1))
