import gymnasium as gym
from stable_baselines3 import PPO # PPO라는 유명한 RL 알고리즘을 가져옴
import time
# 1. 환경(게임)을 만듭니다.
env = gym.make("LunarLander-v3", render_mode="human") # human 모드는 화면에 보여줌

# 2. 알고리즘(AI 에이전트)을 로드합니다.
# MlpPolicy는 "다층 퍼셉트론(신경망)"을 사용하겠다는 의미
model = PPO("MlpPolicy", env, verbose=1, device="cpu")

# 3. 학습을 시작합니다! (총 50,000 타임스텝)
print("----- 학습 시작 -----")
model.learn(total_timesteps=500000)
print("----- 학습 완료 -----")
model.save("ppo_lunarlander_model") # 학습된 모델 저장

# 4. 학습된 모델로 테스트(추론)해보기
print("----- 테스트 시작 -----")
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    time.sleep(0.01)  # <--- 2. 이 코드를 추가해 1/100초씩 멈추게 합니다.

    if terminated or truncated:
        print("----- 미션 종료 (리셋) -----")
        obs, info = env.reset()

env.close()