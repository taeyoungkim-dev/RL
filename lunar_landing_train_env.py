import gymnasium as gym
#PPO algorithm, DQN algorithm
from stable_baselines3 import PPO
from stable_baselines3 import DQN
import time

# --- 1. 학습 (Training) ---
# print("----- 학습용 환경 생성 -----")
train_env = gym.make("LunarLander-v3") 
# model = PPO("MlpPolicy", train_env, verbose=1, device="cpu")
# print("----- 학습 시작 (렌더링 없음, 매우 빠름) -----")
# model.learn(total_timesteps=500000) 
# model.save("ppo_lunarlander_500k")

model = DQN("MlpPolicy", train_env, verbose=1, device="cpu")
model.learn(total_timesteps=500000)
model.save("dqn_lunarlander_500K")
# print("----- 학습 완료 및 모델 저장 -----")


# --- 2. 테스트 (Testing) ---
# 이제 '사람이 볼' 테스트용 환경을 *새로* 만듭니다.
print("----- 테스트용 환경 생성 (human-render) -----")
test_env = gym.make("LunarLander-v3", render_mode="human")

#model = PPO.load("ppo_lunarlander_500k", device="cpu") 
#model = DQN.load("dqn_lunarlander_500K", device="cpu")
obs, info = test_env.reset() # test_env 사용
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action) # test_env 사용
    
    time.sleep(0.01)

    if terminated or truncated:
        print("----- 미션 종료 (리셋) -----")
        obs, info = test_env.reset() # test_env 사용

test_env.close()