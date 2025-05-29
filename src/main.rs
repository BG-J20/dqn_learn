mod agent;
mod env;
mod network;
mod replay_buffer;
mod dqn;

use agent::Agent;
use env::Env;
use network::Network;
use crate::replay_buffer::ReplayBuffer;

fn main() {
    let input_size = 4;     // например, 4 состояния
    let hidden_size = 16;
    let output_size = 2;    // например, 2 действия (влево/вправо)

    let network = Network::new(&[input_size, hidden_size, output_size]);
    let replay_buffer = ReplayBuffer::new(10_000); // или другой размер буфера

    let agent = Agent::new(network, replay_buffer, output_size);

    let env = Env::new();

    let episodes = 100;
    let batch_size = 32;

    dqn::train(agent, env, episodes, batch_size);
}