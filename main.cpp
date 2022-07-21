#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <math.h>       /* sin */
#include <random>
#include "Net.h"

// Device
auto cuda_available = torch::cuda::is_available();
torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
std::default_random_engine generator;

class CustomEnv{
public:
    int min_action;
    int max_action;
    int start;
    int goal;
    int low_state;
    int high_state;
    int num_steps;
    float state;

    CustomEnv(){
        this->min_action = -5;
        this->max_action = 5;
        this->start = 0;
        this->goal = 100;
        this->low_state = -150;
        this->high_state = 150;
        this->num_steps = 0;
        this->state = 0.0;
    }

    std::tuple<float, float, bool> step(float action){
        bool done = 0;
        this->num_steps++;
        this->state += action;

        float reward = action - 6;

        if (state >= goal){
            reward = 20;
            done = true;
        }

        if (num_steps == 30){
            done = true;
        }

        return {state, reward, done};
    }

    float reset(){
        this->num_steps = 0;
        this->state = 0.0;

        return this->state;
    }
};

class Agent{
public:
    float gamma = 0.99;
    int n_outputs = 1;
    int n_actions = 2;
    NeuralNet actor = nullptr;
    NeuralNet critic = nullptr;
    int layer1_size = 64;
    int layer2_size = 64;
    torch::Tensor log_probs;
    torch::optim::Adam *actor_optimizer;
    torch::optim::Adam *critic_optimizer;

    Agent(float alpha, float beta, int input_dims){
        this->actor = NeuralNet(input_dims, layer1_size, n_actions);
        this->actor->to(device);

        critic = NeuralNet(input_dims, layer1_size, 1);
        critic->to(device);

        // Optimizer
        actor_optimizer = new torch::optim::Adam(actor->parameters(), torch::optim::AdamOptions(alpha));
        critic_optimizer = new torch::optim::Adam(critic->parameters(), torch::optim::AdamOptions(beta));
    }

    float choose_action(float observation){
        torch::Tensor logsigma;
        //std::cout<<observation<<std::endl;
        torch::Tensor test = torch::full({1, 1}, /*value=*/observation);
        //std::cout<<test<<std::endl;
        torch::Tensor tensor = torch::ones(5);
        torch::Tensor output = actor->forward(test);
        //std::cout<<output;


        torch::Tensor mu = output[0][0];
        logsigma = output[0][1]; //add exp of sigma
        //std::cout<<mu<<logsigma<<std::endl;
        torch::Tensor sigma = torch::exp(logsigma);
        //std::cout<<logsigma<<sigma<<std::endl;

        std::normal_distribution<float> distribution(mu.item<float>(), sigma.item<float>());

        // auto sampler1 = torch::randn({1}) * sigma + mu ;
        // auto pdf = (1.0 / (sigma * std::sqrt(2.0 * M_PI))) * torch::exp(-0.5 * torch::pow((sampler1 - mu) / sigma, 2));
        // this->log_probs = torch::log(pdf);
        auto sample = torch::randn({1})*sigma + mu;

        // float action = tanh(sampler1.item<float>());
        auto pdf = (1.0 / (sigma * std::sqrt(2*M_PI))) * torch::exp(-0.5 * torch::pow((sample.detach() - mu) / sigma, 2));
        auto probs = distribution(generator);
        //std::cout<<"pdf"<<pdf<<std::endl;
        this->log_probs = torch::log(pdf);
        //this->log_probs = torch::log(torch::tensor(probs));
        //std::cout<<probs<<log_probs<<std::endl;

        float action = tanh(sample.item<float>());

        return action * 5;
    }

    void learn(float state, float reward, float new_state, bool done){
        this->actor_optimizer->zero_grad();
        this->critic_optimizer->zero_grad();

        torch::Tensor critic_value_ = this->critic->forward(torch::full({1,1}, new_state));
        torch::Tensor critic_value = this->critic->forward(torch::full({1,1},state));

        torch::Tensor tensor_reward = torch::tensor(reward);
        torch::Tensor delta = tensor_reward + this->gamma*critic_value_ * (1 * int(!done)) - critic_value;

        auto actor_loss = -1 * this->log_probs * delta;
        auto critic_loss = torch::pow(delta, 2);
        //std::cout<<log_probs<<std::endl<<delta<<std::endl;

        (actor_loss + critic_loss).backward();
        this->actor_optimizer->step();
        this->critic_optimizer->step();
    }
};

int main() {
    std::cout << "FeedForward Neural Network\n\n";

    // Hyper parameters
    const int64_t input_size = 1;
    const int64_t hidden_size = 256;
    const int64_t num_classes = 2;
    const int64_t batch_size = 100;
    const size_t num_epochs = 5;
    const double alpha = 0.000005;
    const double beta = 0.00001;

    CustomEnv env;

    Agent *agent = new Agent(0.0000005, 0.000001, 1);

    int num_episodes = 800000;
    bool done;
    float score;
    float observation, observation_, reward, action;

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (int i = 0; i < num_episodes; i++){
        done = false;
        score = 0;
        observation = env.reset();

        while (!done){
            action = agent->choose_action(observation);
            std::tie(observation_, reward, done) = env.step(action);
            agent->learn(observation, reward, observation_, done);
            observation = observation_;
            score += reward;
        }

        printf("Episode %d score %.2f\n", i, score);
    }

    return 0;
}