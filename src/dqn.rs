use crate::agent::Agent;
use crate::env::*;
use crate::replay_buffer::Experience;

use std::fs::File;
use std::io::Write;

pub fn train(mut agent: Agent, mut env: Env, episodes: usize, batch_size: usize) {
    for episode in 0..episodes {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut steps = 0;

        while !env.is_done() {
            //agent select actions
            let action = agent.select_action(state.clone());

            //perform a ste[ in the environment
            let (next_state, reward, done) = env.step(action);

            //save exp
            let experience = Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            };
            agent.remember(experience);

            if agent.can_learn(batch_size) {
                agent.learn(batch_size);
            }

            state = next_state;
            total_reward += reward;
            steps += 1;
        }
        println!("Episode {} | Total reward: {:.2} | Steps: {}", episode, total_reward, steps);

        if episode % 10 == 0 {
            save_model(&agent, &format!("model_ep_{}.bin", episode));
        }

    }
}

fn save_model(agent: &Agent, filename: &str) {
    let serialized = bincode::serialize(&agent.network).unwrap();
    let mut file = File::create(filename).unwrap();
    file.write_all(&serialized).unwrap();
}